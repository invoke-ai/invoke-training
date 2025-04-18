import itertools
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from peft import PeftModel
from typing import Literal, Optional, Union
from torchvision import transforms

import peft
import torch
import torch.utils.data
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPPreTrainedModel, CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from invoke_training._shared.accelerator.accelerator_utils import (
    get_dtype_from_str,
    initialize_accelerator,
    initialize_logging,
)

from invoke_training._shared.checkpoints.checkpoint_tracker import CheckpointTracker
from invoke_training._shared.data.data_loaders.dreambooth_sd_dataloader import build_dreambooth_sd_dataloader
from invoke_training._shared.data.data_loaders.image_caption_flux_dataloader import build_image_caption_flux_dataloader
from invoke_training._shared.data.samplers.aspect_ratio_bucket_batch_sampler import log_aspect_ratio_buckets
from invoke_training._shared.data.transforms.tensor_disk_cache import TensorDiskCache
from invoke_training._shared.optimizer.optimizer_utils import initialize_optimizer
from invoke_training._shared.flux.lora_checkpoint_utils import (
    save_flux_peft_checkpoint,
)
from invoke_training._shared.flux.encoding_utils import encode_prompt
from invoke_training._shared.flux.model_loading_utils import load_models_flux
from invoke_training._shared.flux.validation import generate_validation_images_flux
from invoke_training._shared.stable_diffusion.tokenize_captions import tokenize_captions

from invoke_training.config.data.data_loader_config import ImageCaptionSDDataLoaderConfig
from invoke_training.pipelines.callbacks import ModelCheckpoint, ModelType, PipelineCallbacks, TrainingCheckpoint
from invoke_training.pipelines.flux.lora.config import FluxLoraConfig


def _save_flux_lora_checkpoint(
    epoch: int,
    step: int,
    transformer: peft.PeftModel | None,
    text_encoder_1: CLIPTextModel | None,
    text_encoder_2: T5EncoderModel | None,
    logger: logging.Logger,
    checkpoint_tracker: CheckpointTracker,
    callbacks: list[PipelineCallbacks] | None,
    lora_checkpoint_format: Literal["invoke_peft", "kohya"] = "invoke_peft",
):
    # Prune checkpoints and get new checkpoint path.
    num_pruned = checkpoint_tracker.prune(1)
    if num_pruned > 0:
        logger.info(f"Pruned {num_pruned} checkpoint(s).")
    save_path = checkpoint_tracker.get_path(epoch=epoch, step=step)

    if lora_checkpoint_format == "invoke_peft":
        model_type = ModelType.FLUX_LORA_PEFT
        save_flux_peft_checkpoint(
            Path(save_path), transformer=transformer, text_encoder_1=text_encoder_1, text_encoder_2=text_encoder_2
        )
    else:
        raise ValueError(f"Unsupported lora_checkpoint_format: '{lora_checkpoint_format}'.")

    if callbacks is not None:
        for cb in callbacks:
            cb.on_save_checkpoint(
                TrainingCheckpoint(
                    models=[ModelCheckpoint(file_path=save_path, model_type=model_type)], epoch=epoch, step=step
                )
            )


def _build_data_loader(
    data_loader_config: Union[ImageCaptionSDDataLoaderConfig],
    batch_size: int,
    use_masks: bool = False,
    text_encoder_output_cache_dir: Optional[str] = None,
    vae_output_cache_dir: Optional[str] = None,
    shuffle: bool = True,
    sequential_batching: bool = False,
) -> DataLoader:
    if data_loader_config.type == "IMAGE_CAPTION_FLUX_DATA_LOADER":
        return build_image_caption_flux_dataloader(
            config=data_loader_config,
            batch_size=batch_size,
            use_masks=use_masks,
            text_encoder_output_cache_dir=text_encoder_output_cache_dir,
            text_encoder_cache_field_to_output_field={"text_encoder_output": "text_encoder_output"},
            vae_output_cache_dir=vae_output_cache_dir,
            shuffle=shuffle,
        )
    else:
        raise ValueError(f"Unsupported data loader config type: '{data_loader_config.type}'.")


def cache_text_encoder_outputs(
    cache_dir: str, config: FluxLoraConfig, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel
):
    """Run the text encoder on all captions in the dataset and cache the results to disk.

    Args:
        cache_dir (str): The directory where the results will be cached.
        config (FluxLoraConfig): Training config.
        tokenizer (CLIPTokenizer): The tokenizer.
        text_encoder (CLIPTextModel): The text_encoder.
    """
    data_loader = _build_data_loader(
        data_loader_config=config.data_loader,
        batch_size=config.train_batch_size,
        shuffle=False,
        sequential_batching=True,
    )

    cache = TensorDiskCache(cache_dir)

    for data_batch in tqdm(data_loader):
        caption_token_ids = tokenize_captions(tokenizer, data_batch["caption"]).to(text_encoder.device)
        text_encoder_output_batch = text_encoder(caption_token_ids)[0]
        # Split batch before caching.
        for i in range(len(data_batch["id"])):
            cache.save(data_batch["id"][i], {"text_encoder_output": text_encoder_output_batch[i]})


def cache_vae_outputs(cache_dir: str, data_loader: DataLoader, vae: AutoencoderKL):
    """Run the VAE on all images in the dataset and cache the results to disk."""
    cache = TensorDiskCache(cache_dir)

    for data_batch in tqdm(data_loader):
        latents = vae.encode(data_batch["image"].to(device=vae.device, dtype=vae.dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        # Split batch before caching.
        for i in range(len(data_batch["id"])):
            data = {
                "vae_output": latents[i],
                "original_size_hw": data_batch["original_size_hw"][i],
                "crop_top_left_yx": data_batch["crop_top_left_yx"][i],
            }
            if "mask" in data_batch:
                data["mask"] = data_batch["mask"][i]
            cache.save(data_batch["id"][i], data)

def get_noisy_latents(noise_scheduler: FlowMatchEulerDiscreteScheduler, 
                    latents: torch.Tensor, 
                    config: FluxLoraConfig):
    """ 
    Generate random noise. Sample a random timestep from the distribution chosen by the config. 
    Linearly interpolate between the latents and the noise based on timestep.
    See Section 3.1 of https://arxiv.org/pdf/2403.03206v1 for timestep sampling.

    Args:
        noise_scheduler (FlowMatchEulerDiscreteScheduler): The noise scheduler.
        latents (torch.Tensor): The latents.
        config (FluxLoraConfig): The config.

    Returns:
        torch.Tensor: The noisy latents.

    """

    batch_size = latents.shape[0]
    dtype = latents.dtype
    device = latents.device
    noise = torch.randn_like(latents)

    if config.timestep_sampler == "shift":
        shift = config.discrete_flow_shift
        sigmas = torch.randn(batch_size, device=device)
        sigmas = sigmas * config.sigmoid_scale  # larger scale for more uniform sampling
        sigmas = sigmas.sigmoid()
        sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * noise_scheduler.config.num_train_timesteps
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=device)
        sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)
    
    sigmas = sigmas.view(-1, 1, 1, 1)

    # Linearly interpolate between the latents and the noise.
    noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
    return noisy_model_input.to(dtype), noise.to(dtype), timesteps.to(dtype), sigmas.to(dtype)

def decode_latents(vae: AutoencoderKL, latents: torch.Tensor):
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents).sample

    # tensor to image
    image = image.cpu().numpy()
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)

    image.save("image.png")
    return image

def train_forward(  # noqa: C901
    config: FluxLoraConfig,
    data_batch: dict,
    vae: AutoencoderKL,
    noise_scheduler: FlowMatchEulerDiscreteScheduler,
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: T5Tokenizer,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: T5EncoderModel,
    transformer: FluxTransformer2DModel | PeftModel,
    weight_dtype: torch.dtype,
    use_masks: bool = False,
    min_snr_gamma: float | None = None,
    logger: logging.Logger = None,
) -> torch.Tensor:
    """Run the forward training pass for a single data_batch.

    Returns:
        torch.Tensor: Loss
    """
    # Convert images to latent space.
    # The VAE output may have been cached and included in the data_batch. If not, we calculate it here.
    latents = data_batch.get("vae_output", None)
    if latents is None:
        # Cast input image to same dtype as VAE
        image = data_batch["image"].to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(image).latent_dist.sample()
        batch_size, num_channels, height, width = latents.shape
        latents = latents * vae.config.scaling_factor
        latents = FluxPipeline._pack_latents(latents, batch_size, num_channels, height, width)
    else:
        batch_size, num_channels, height, width = latents.shape
    # Sample noise that we'll add to the latents.
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
        batch_size, height // 2, width // 2, latents.device, latents.dtype
    )

    # Add noise to the latents according to the noise magnitude at each timestep (this is the forward
    # diffusion process).
    noisy_latents, noise, timesteps, sigmas = get_noisy_latents(noise_scheduler, latents, config)
        
    # Get the text embedding for conditioning.
    # The text encoder output may have been cached and included in the data_batch. If not, we calculate it here.
    if "prompt_embeds" in data_batch:
        prompt_embeds = data_batch["prompt_embeds"]
        pooled_prompt_embeds = data_batch["pooled_prompt_embeds"]
    else:
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            prompt=data_batch["caption"],
            prompt_2=data_batch.get("caption_2", None),
            clip_tokenizer=tokenizer_1,
            t5_tokenizer=tokenizer_2,
            clip_text_encoder=text_encoder_1,
            t5_text_encoder=text_encoder_2,
            device=latents.device,
            num_images_per_prompt=1,
            lora_scale=config.lora_scale,
            clip_tokenizer_max_length=config.clip_tokenizer_max_length,
            t5_tokenizer_max_length=config.t5_tokenizer_max_length,
            logger=logger
        )
        
    guidance = torch.full((batch_size,), float(config.guidance_scale), device=latents.device)
    model_pred = transformer(hidden_states=noisy_latents[0],
                            timestep=timesteps / 1000,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            guidance=guidance,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            return_dict=False,
                        )[0]
    ### Flow matching loss
    # See here for more discussion:https://discuss.huggingface.co/t/meaning-of-vector-fields-in-flux-and-sd3-loss-function/106601
    target = noise - latents

    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
    loss = loss.mean(dim=list(range(1, len(loss.shape))))
    return loss.mean()


def train(config: FluxLoraConfig, callbacks: list[PipelineCallbacks] | None = None):  # noqa: C901

    # Create a timestamped directory for all outputs.
    out_dir = os.path.join(config.base_output_dir, f"{time.time()}")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir)

    accelerator = initialize_accelerator(
        out_dir, config.gradient_accumulation_steps, config.mixed_precision, config.report_to
    )
    logger = initialize_logging(os.path.basename(__file__), accelerator)

    # Set the accelerate seed.
    if config.seed is not None:
        set_seed(config.seed)

    # Log the accelerator configuration from every process to help with debugging.
    logger.info(accelerator.state, main_process_only=False)

    logger.info("Starting LoRA Training.")
    logger.info(f"Configuration:\n{json.dumps(config.dict(), indent=2, default=str)}")
    logger.info(f"Output dir: '{out_dir}'")

    # Write the configuration to disk.
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config.dict(), f, indent=2, default=str)

    weight_dtype = get_dtype_from_str(config.weight_dtype)

    logger.info("Loading models.")
    tokenizer_1, tokenizer_2, noise_scheduler, text_encoder_1, text_encoder_2, vae, transformer = load_models_flux(
        model_name_or_path=config.model,
        transformer_path=config.transformer_path,
        text_encoder_1_path=config.text_encoder_1_path,
        text_encoder_2_path=config.text_encoder_2_path,
        dtype=weight_dtype,
        logger=logger,
    )

    # Prepare text encoder output cache.
    text_encoder_output_cache_dir_name = None
    if config.cache_text_encoder_outputs:
        # TODO(ryand): Think about how to better check if it is safe to cache the text encoder outputs. Currently, there
        # are a number of configurations that would cause variation in the text encoder outputs and should not be used
        # with caching.

        # We use a temporary directory for the cache. The directory will automatically be cleaned up when
        # tmp_text_encoder_output_cache_dir is destroyed.
        tmp_text_encoder_output_cache_dir = tempfile.TemporaryDirectory()
        text_encoder_output_cache_dir_name = tmp_text_encoder_output_cache_dir.name
        if accelerator.is_local_main_process:
            # Only the main process should populate the cache.
            logger.info(f"Generating text encoder output cache ('{text_encoder_output_cache_dir_name}').")
            text_encoder_1.to(accelerator.device, dtype=weight_dtype)
            text_encoder_2.to(accelerator.device, dtype=weight_dtype)
            # TODO(ryan): Move cache_text_encoder_outputs to a shared location so that it is not imported from another
            # pipeline.
            cache_text_encoder_outputs(
                text_encoder_output_cache_dir_name, config, tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2
            )
        # Move the text_encoders back to the CPU, because they are not needed for training.
        text_encoder_1.to("cpu")
        text_encoder_2.to("cpu")
        accelerator.wait_for_everyone()
    else:
        text_encoder_1.to(accelerator.device, dtype=weight_dtype)
        text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # Prepare VAE output cache.
    # vae_output_cache_dir_name = None
    if config.cache_vae_outputs:
        if config.data_loader.random_flip:
            raise ValueError("'cache_vae_outputs' cannot be True if 'random_flip' is True.")
        if not config.data_loader.center_crop:
            raise ValueError("'cache_vae_outputs' cannot be True if 'center_crop' is False.")

        # We use a temporary directory for the cache. The directory will automatically be cleaned up when
        # tmp_vae_output_cache_dir is destroyed.
        tmp_vae_output_cache_dir = tempfile.TemporaryDirectory()
        vae_output_cache_dir_name = tmp_vae_output_cache_dir.name
        if accelerator.is_local_main_process:
            # Only the main process should populate the cache.
            logger.info(f"Generating VAE output cache ('{vae_output_cache_dir_name}').")
            vae.to(accelerator.device, dtype=weight_dtype)
            data_loader = _build_data_loader(
                data_loader_config=config.data_loader,
                batch_size=config.train_batch_size,
                use_masks=config.use_masks,
                shuffle=False,
                sequential_batching=True,
            )
            cache_vae_outputs(vae_output_cache_dir_name, data_loader, vae)
        # Move the VAE back to the CPU, because it is not needed for training.
        vae.to("cpu")
        accelerator.wait_for_everyone()
    else:
        vae.to(accelerator.device, dtype=weight_dtype)

    transformer.to(accelerator.device, dtype=weight_dtype)

    # Add LoRA layers to the models being trained.
    trainable_param_groups = []
    all_trainable_models: list[peft.PeftModel] = []

    def inject_lora_layers(model, lora_config: peft.LoraConfig, lr: float | None = None) -> peft.PeftModel:
        peft_model = peft.get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()

        # Populate `trainable_param_groups`, to be passed to the optimizer.
        param_group = {"params": list(filter(lambda p: p.requires_grad, peft_model.parameters()))}
        if lr is not None:
            param_group["lr"] = lr
        trainable_param_groups.append(param_group)

        # Populate all_trainable_models.
        all_trainable_models.append(peft_model)

        peft_model.train()

        return peft_model

    # Add LoRA layers to the model.
    if config.train_transformer:
        transformer_lora_config = peft.LoraConfig(
            r=config.lora_rank_dim,
            # TODO(ryand): Diffusers uses lora_alpha=config.lora_rank_dim. Is that preferred?
            lora_alpha=1.0,
            target_modules=config.flux_lora_target_modules,
        )
        transformer = inject_lora_layers(transformer, transformer_lora_config, lr=config.transformer_learning_rate)

    if config.train_text_encoder:
        text_encoder_lora_config = peft.LoraConfig(
            r=config.lora_rank_dim,
            lora_alpha=1.0,
            # init_lora_weights="gaussian",
            target_modules=config.text_encoder_lora_target_modules,
        )
        text_encoder_1 = inject_lora_layers(
            text_encoder_1, text_encoder_lora_config, lr=config.text_encoder_learning_rate
        )
        text_encoder_2 = inject_lora_layers(
            text_encoder_2, text_encoder_lora_config, lr=config.text_encoder_learning_rate
        )

    # Enable gradient checkpointing.
    if config.gradient_checkpointing:
        # We want to enable gradient checkpointing in the UNet regardless of whether it is being trained.
        transformer.enable_gradient_checkpointing()
        # unet must be in train() mode for gradient checkpointing to take effect.
        # At the time of writing, the unet dropout probabilities default to 0, so putting the unet in train mode does
        # not change its forward behavior.
        transformer.train()
        if config.train_text_encoder:
            text_encoder_1.gradient_checkpointing_enable()
            text_encoder_2.gradient_checkpointing_enable()
            # The text encoders must be in train() mode for gradient checkpointing to take effect. This should
            # already be the case, since we are training the text_encoders, be we do it explicitly to make it clear
            # that this is required.
            # At the time of writing, the text encoder dropout probabilities default to 0, so putting the text
            # encoders in train mode does not change their forward behavior.
            text_encoder_1.train()
            text_encoder_2.train()
            # Set requires_grad = True on the first parameters of the text encoders. Without this, the text encoder
            # LoRA weights would have 0 gradients, and so would not get trained. Note that the set of
            # trainable_param_groups has already been populated - the embeddings will not be trained.
            text_encoder_1.text_model.embeddings.requires_grad_(True)
            text_encoder_2.text_model.embeddings.requires_grad_(True)
    optimizer = initialize_optimizer(config.optimizer, trainable_param_groups)

    data_loader = _build_data_loader(
        data_loader_config=config.data_loader,
        batch_size=config.train_batch_size,
        use_masks=config.use_masks,
        # text_encoder_output_cache_dir=text_encoder_output_cache_dir_name,
        # vae_output_cache_dir=vae_output_cache_dir_name,
    )

    # log_aspect_ratio_buckets(logger=logger, batch_sampler=data_loader.batch_sampler)

    assert sum([config.max_train_steps is not None, config.max_train_epochs is not None]) == 1
    assert sum([config.save_every_n_steps is not None, config.save_every_n_epochs is not None]) == 1
    assert sum([config.validate_every_n_steps is not None, config.validate_every_n_epochs is not None]) == 1

    # A "step" represents a single weight update operation (i.e. takes into account gradient accumulation steps).
    # math.ceil(...) is used in calculating the num_steps_per_epoch, because by default an optimizer step is taken when
    # the end of the dataloader is reached, even if gradient_accumulation_steps hasn't been reached.
    num_steps_per_epoch = math.ceil(len(data_loader) / config.gradient_accumulation_steps)
    num_train_steps = config.max_train_steps or config.max_train_epochs * num_steps_per_epoch
    num_train_epochs = math.ceil(num_train_steps / num_steps_per_epoch)

    # TODO(ryand): Test in a distributed training environment and more clearly document the rationale for scaling steps
    # by the number of processes. This scaling logic was copied from the diffusers example training code, but it appears
    # in many places so I don't know where it originated. Internally, accelerate makes one LR scheduler step per process
    # (https://github.com/huggingface/accelerate/blame/49cb83a423f2946059117d8bb39b7c8747d29d80/src/accelerate/scheduler.py#L72-L82),
    # so the scaling here simply reverses that behaviour.
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_train_steps * accelerator.num_processes,
    )

    prepared_result: tuple[
        FluxTransformer2DModel,
        CLIPTextModel,
        T5EncoderModel,
        torch.optim.Optimizer,
        torch.utils.data.DataLoader,
        torch.optim.lr_scheduler.LRScheduler,
    ] = accelerator.prepare(
        transformer,
        text_encoder_1,
        text_encoder_2,
        optimizer,
        data_loader,
        lr_scheduler,
        # Disable automatic device placement for text_encoder if the text encoder outputs were cached.
        device_placement=[
            True,
            not config.cache_text_encoder_outputs,
            not config.cache_text_encoder_outputs,
            True,
            True,
            True,
        ],
    )
    transformer, text_encoder_1, text_encoder_2, optimizer, data_loader, lr_scheduler = prepared_result

    if accelerator.is_main_process:
        accelerator.init_trackers("lora_training")
        # Tensorboard uses markdown formatting, so we wrap the config json in a code block.
        accelerator.log({"configuration": f"```json\n{json.dumps(config.dict(), indent=2, default=str)}\n```\n"})

    checkpoint_tracker = CheckpointTracker(
        base_dir=ckpt_dir,
        prefix="checkpoint",
        max_checkpoints=config.max_checkpoints,
        extension=".safetensors" if config.lora_checkpoint_format == "kohya" else None,
    )

    # Train!
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num batches = {len(data_loader)}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Parallel processes = {accelerator.num_processes}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {num_train_steps}")
    logger.info(f"  Total epochs = {num_train_epochs}")

    global_step = 0
    first_epoch = 0
    completed_epochs = 0

    progress_bar = tqdm(
        range(global_step, num_train_steps),
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    def save_checkpoint(num_completed_epochs: int, num_completed_steps: int):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            _save_flux_lora_checkpoint(
                epoch=num_completed_epochs,
                step=num_completed_steps,
                transformer=transformer if config.train_transformer else None,
                text_encoder_1=text_encoder_1 if config.train_text_encoder else None,
                text_encoder_2=text_encoder_2 if config.train_text_encoder else None,
                logger=logger,
                checkpoint_tracker=checkpoint_tracker,
                lora_checkpoint_format=config.lora_checkpoint_format,
                callbacks=callbacks,
            )
        accelerator.wait_for_everyone()

    def validate(num_completed_epochs: int, num_completed_steps: int):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            generate_validation_images_flux(
                epoch=num_completed_epochs,
                step=num_completed_steps,
                out_dir=out_dir,
                accelerator=accelerator,
                vae=vae,
                text_encoder_1=text_encoder_1,
                text_encoder_2=text_encoder_2,
                tokenizer_1=tokenizer_1,
                tokenizer_2=tokenizer_2,
                noise_scheduler=noise_scheduler,
                transformer=transformer,
                config=config,
                logger=logger,
                callbacks=callbacks,
            )
        accelerator.wait_for_everyone()

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for data_batch_idx, data_batch in enumerate(data_loader):
            # (Pdb) data_batch['image'].shape
            # torch.Size([4, 3, 512, 512])
            with accelerator.accumulate(transformer, text_encoder_1, text_encoder_2):
                loss = train_forward(
                    config=config,
                    data_batch=data_batch,
                    vae=vae,
                    noise_scheduler=noise_scheduler,
                    tokenizer_1=tokenizer_1,
                    tokenizer_2=tokenizer_2,
                    text_encoder_1=text_encoder_1,
                    text_encoder_2=text_encoder_2,
                    transformer=transformer,
                    weight_dtype=weight_dtype,
                    use_masks=config.use_masks,
                    min_snr_gamma=config.min_snr_gamma,
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                # TODO(ryand): Test that this works properly with distributed training.
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps

                # Backpropagate.
                accelerator.backward(loss)
                if accelerator.sync_gradients and config.max_grad_norm is not None:
                    params_to_clip = itertools.chain.from_iterable([m.parameters() for m in all_trainable_models])
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes.
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                completed_epochs = epoch if (data_batch_idx + 1) < len(data_loader) else epoch + 1
                log = {"train_loss": train_loss}

                lrs = lr_scheduler.get_last_lr()
                if config.train_transformer:
                    # When training the UNet, it will always be the first parameter group.
                    log["lr/transformer"] = float(lrs[0])
                    if config.optimizer.optimizer_type == "Prodigy":
                        log["lr/d*lr/transformer"] = optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
                if config.train_text_encoder:
                    # When training the text encoder, it will always be the last parameter group.
                    log["lr/text_encoder"] = float(lrs[-1])
                    if config.optimizer.optimizer_type == "Prodigy":
                        log["lr/d*lr/text_encoder"] = optimizer.param_groups[-1]["d"] * optimizer.param_groups[-1]["lr"]

                accelerator.log(log, step=global_step)
                train_loss = 0.0

                # global_step represents the *number of completed steps* at this point.
                if config.save_every_n_steps is not None and global_step % config.save_every_n_steps == 0:
                    save_checkpoint(num_completed_epochs=completed_epochs, num_completed_steps=global_step)

                if (
                    config.validate_every_n_steps is not None
                    and global_step % config.validate_every_n_steps == 0
                    and len(config.validation_prompts) > 0
                ):
                    validate(num_completed_epochs=completed_epochs, num_completed_steps=global_step)

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= num_train_steps:
                break

        # Save a checkpoint every n epochs.
        if config.save_every_n_epochs is not None and completed_epochs % config.save_every_n_epochs == 0:
            save_checkpoint(num_completed_epochs=completed_epochs, num_completed_steps=global_step)

        # Generate validation images every n epochs.
        if (
            config.validate_every_n_epochs is not None
            and completed_epochs % config.validate_every_n_epochs == 0
            and len(config.validation_prompts) > 0
        ):
            validate(num_completed_epochs=completed_epochs, num_completed_steps=global_step)

    accelerator.end_training()

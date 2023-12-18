import json
import logging
import math
import os
import tempfile
import time
from typing import Optional, Union

import numpy as np
import torch
import torch.utils.data
from accelerate import Accelerator
from accelerate.hooks import remove_hook_from_module
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from invoke_training.config.pipelines.finetune_lora_config import FinetuneLoRASDConfig
from invoke_training.config.shared.data.data_loader_config import (
    DreamboothSDDataLoaderConfig,
    ImageCaptionSDDataLoaderConfig,
)
from invoke_training.core.lora.injection.stable_diffusion import (
    inject_lora_into_clip_text_encoder,
    inject_lora_into_unet,
)
from invoke_training.training.shared.accelerator.accelerator_utils import (
    get_mixed_precision_dtype,
    initialize_accelerator,
    initialize_logging,
)
from invoke_training.training.shared.checkpoints.checkpoint_tracker import CheckpointTracker
from invoke_training.training.shared.data.data_loaders.dreambooth_sd_dataloader import build_dreambooth_sd_dataloader
from invoke_training.training.shared.data.data_loaders.image_caption_sd_dataloader import (
    build_image_caption_sd_dataloader,
)
from invoke_training.training.shared.data.transforms.tensor_disk_cache import TensorDiskCache
from invoke_training.training.shared.optimizer.optimizer_utils import initialize_optimizer
from invoke_training.training.shared.stable_diffusion.lora_checkpoint_utils import save_lora_checkpoint
from invoke_training.training.shared.stable_diffusion.model_loading_utils import PipelineVersionEnum, load_pipeline
from invoke_training.training.shared.stable_diffusion.tokenize_captions import tokenize_captions


def load_models(
    config: FinetuneLoRASDConfig,
) -> tuple[CLIPTokenizer, DDPMScheduler, CLIPTextModel, AutoencoderKL, UNet2DConditionModel]:
    """Load all models required for training from disk, transfer them to the
    target training device and cast their weight dtypes.

    Args:
        config (FinetuneLoRASDConfig): The LoRA training run config.
        logger (logging.Logger): A logger.

    Returns:
        tuple[
            CLIPTokenizer,
            DDPMScheduler,
            CLIPTextModel,
            AutoencoderKL,
            UNet2DConditionModel,
        ]: A tuple of loaded models.
    """
    pipeline: StableDiffusionPipeline = load_pipeline(config.model, PipelineVersionEnum.SD)

    # Extract sub-models from the pipeline.
    tokenizer: CLIPTokenizer = pipeline.tokenizer
    text_encoder: CLIPTextModel = pipeline.text_encoder
    vae: AutoencoderKL = pipeline.vae
    unet: UNet2DConditionModel = pipeline.unet
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
        steps_offset=1,
    )

    # Disable gradient calculation for model weights to save memory.
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Put models in 'eval' mode.
    text_encoder.eval()
    vae.eval()
    unet.eval()

    return tokenizer, noise_scheduler, text_encoder, vae, unet


def build_data_loader(
    data_loader_config: Union[ImageCaptionSDDataLoaderConfig, DreamboothSDDataLoaderConfig],
    batch_size: int,
    text_encoder_output_cache_dir: Optional[str] = None,
    vae_output_cache_dir: Optional[str] = None,
    shuffle: bool = True,
    sequential_batching: bool = False,
) -> DataLoader:
    if data_loader_config.type == "IMAGE_CAPTION_SD_DATA_LOADER":
        return build_image_caption_sd_dataloader(
            config=data_loader_config,
            batch_size=batch_size,
            text_encoder_output_cache_dir=text_encoder_output_cache_dir,
            text_encoder_cache_field_to_output_field={"text_encoder_output": "text_encoder_output"},
            vae_output_cache_dir=vae_output_cache_dir,
            shuffle=shuffle,
        )
    elif data_loader_config.type == "DREAMBOOTH_SD_DATA_LOADER":
        return build_dreambooth_sd_dataloader(
            data_loader_config=data_loader_config,
            batch_size=batch_size,
            text_encoder_output_cache_dir=text_encoder_output_cache_dir,
            vae_output_cache_dir=vae_output_cache_dir,
            shuffle=shuffle,
            sequential_batching=sequential_batching,
        )
    else:
        raise ValueError(f"Unsupported data loader config type: '{data_loader_config.type}'.")


def cache_text_encoder_outputs(
    cache_dir: str, config: FinetuneLoRASDConfig, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel
):
    """Run the text encoder on all captions in the dataset and cache the results to disk.

    Args:
        cache_dir (str): The directory where the results will be cached.
        config (FinetuneLoRASDConfig): Training config.
        tokenizer (CLIPTokenizer): The tokenizer.
        text_encoder (CLIPTextModel): The text_encoder.
    """
    data_loader = build_data_loader(
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
            cache.save(data_batch["id"][i], {"vae_output": latents[i]})


def generate_validation_images(
    epoch: int,
    out_dir: str,
    accelerator: Accelerator,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    noise_scheduler: DDPMScheduler,
    unet: UNet2DConditionModel,
    config: FinetuneLoRASDConfig,
    logger: logging.Logger,
):
    """Generate validation images for the purpose of tracking image generation behaviour on fixed prompts throughout
    training.

    Args:
        epoch (int): Epoch number, for reporting purposes.
        out_dir (str): The output directory where the validation images will be stored.
        accelerator (Accelerator): Accelerator
        vae (AutoencoderKL):
        text_encoder (CLIPTextModel):
        tokenizer (CLIPTokenizer):
        noise_scheduler (DDPMScheduler):
        unet (UNet2DConditionModel):
        config (FinetuneLoRASDConfig): Training configs.
        logger (logging.Logger): Logger.
    """
    logger.info("Generating validation images.")

    # Record original model devices so that we can restore this state after running the pipeline with CPU model
    # offloading.
    unet_device = unet.device
    vae_device = vae.device
    text_encoder_device = text_encoder.device

    # Create pipeline.
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
        # TODO(ryand): Add safety checker support.
        requires_safety_checker=False,
    )
    if config.enable_cpu_offload_during_validation:
        pipeline.enable_model_cpu_offload(accelerator.device.index or 0)
    else:
        pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # Run inference.
    with torch.no_grad():
        for prompt_idx, prompt in enumerate(config.validation_prompts):
            generator = torch.Generator(device=accelerator.device)
            if config.seed is not None:
                generator = generator.manual_seed(config.seed)

            images = []
            for _ in range(config.num_validation_images_per_prompt):
                with accelerator.autocast():
                    images.append(
                        pipeline(
                            prompt,
                            num_inference_steps=30,
                            generator=generator,
                            height=config.data_loader.image_transforms.resolution,
                            width=config.data_loader.image_transforms.resolution,
                        ).images[0]
                    )

            # Save images to disk.
            validation_dir = os.path.join(
                out_dir,
                "validation",
                f"epoch_{epoch:0>8}",
                f"prompt_{prompt_idx:0>4}",
            )
            os.makedirs(validation_dir)
            for image_idx, image in enumerate(images):
                image.save(os.path.join(validation_dir, f"{image_idx:0>4}.jpg"))

            # Log images to trackers. Currently, only tensorboard is supported.
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(
                        f"validation (prompt {prompt_idx})",
                        np_images,
                        epoch,
                        dataformats="NHWC",
                    )

    del pipeline
    torch.cuda.empty_cache()

    # Remove hooks from models.
    # HACK(ryand): Hooks get added when calling `pipeline.enable_model_cpu_offload(...)`, but `StableDiffusionPipeline`
    # does not offer a way to clean them up so we have to do this manually.
    for model in [unet, vae, text_encoder]:
        remove_hook_from_module(model)

    # Restore models to original devices.
    unet.to(unet_device)
    vae.to(vae_device)
    text_encoder.to(text_encoder_device)


def train_forward(
    config: FinetuneLoRASDConfig,
    data_batch: dict,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    unet: UNet2DConditionModel,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    """Run the forward training pass for a single data_batch.

    Returns:
        torch.Tensor: Loss
    """
    # Convert images to latent space.
    # The VAE output may have been cached and included in the data_batch. If not, we calculate it here.
    latents = data_batch.get("vae_output", None)
    if latents is None:
        latents = vae.encode(data_batch["image"].to(dtype=weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # Sample noise that we'll add to the latents.
    noise = torch.randn_like(latents)

    batch_size = latents.shape[0]
    # Sample a random timestep for each image.
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (batch_size,),
        device=latents.device,
    )
    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep (this is the forward
    # diffusion process).
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Get the text embedding for conditioning.
    # The text_encoder_output may have been cached and included in the data_batch. If not, we calculate it here.
    encoder_hidden_states = data_batch.get("text_encoder_output", None)
    if encoder_hidden_states is None:
        caption_token_ids = tokenize_captions(tokenizer, data_batch["caption"]).to(text_encoder.device)
        encoder_hidden_states = text_encoder(caption_token_ids)[0].to(dtype=weight_dtype)

    # Get the target for loss depending on the prediction type.
    if config.prediction_type is not None:
        # Set the prediction_type of scheduler if it's defined in config.
        noise_scheduler.register_to_config(prediction_type=config.prediction_type)
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    # Predict the noise residual.
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
    if "loss_weight" in data_batch:
        # Mean-reduce the loss along all dimensions except for the batch dimension.
        loss = loss.mean([1, 2, 3])
        # Apply per-example weights.
        loss = loss * data_batch["loss_weight"]
    return loss.mean()


def run_training(config: FinetuneLoRASDConfig):  # noqa: C901
    # Give a clear error message if an unsupported base model was chosen.
    # TODO(ryan): Update this check to work with single-file SD checkpoints.
    # check_base_model_version(
    #     {BaseModelVersionEnum.STABLE_DIFFUSION_V1, BaseModelVersionEnum.STABLE_DIFFUSION_V2},
    #     config.model,
    #     local_files_only=False,
    # )

    # Create a timestamped directory for all outputs.
    out_dir = os.path.join(config.output.base_output_dir, f"{time.time()}")
    os.makedirs(out_dir)

    accelerator = initialize_accelerator(
        out_dir, config.gradient_accumulation_steps, config.mixed_precision, config.output.report_to
    )
    logger = initialize_logging(__name__, accelerator)

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

    weight_dtype = get_mixed_precision_dtype(accelerator)

    logger.info("Loading models.")
    tokenizer, noise_scheduler, text_encoder, vae, unet = load_models(config)

    if config.xformers:
        import xformers  # noqa: F401

        unet.enable_xformers_memory_efficient_attention()
        vae.enable_xformers_memory_efficient_attention()

    # Prepare text encoder output cache.
    text_encoder_output_cache_dir_name = None
    if config.cache_text_encoder_outputs:
        # TODO(ryand): Think about how to better check if it is safe to cache the text encoder outputs. Currently, there
        # are a number of configurations that would cause variation in the text encoder outputs and should not be used
        # with caching.
        if config.train_text_encoder:
            raise ValueError("'cache_text_encoder_outputs' and 'train_text_encoder' cannot both be True.")

        # We use a temporary directory for the cache. The directory will automatically be cleaned up when
        # tmp_text_encoder_output_cache_dir is destroyed.
        tmp_text_encoder_output_cache_dir = tempfile.TemporaryDirectory()
        text_encoder_output_cache_dir_name = tmp_text_encoder_output_cache_dir.name
        if accelerator.is_local_main_process:
            # Only the main process should populate the cache.
            logger.info(f"Generating text encoder output cache ('{text_encoder_output_cache_dir_name}').")
            text_encoder.to(accelerator.device, dtype=weight_dtype)
            cache_text_encoder_outputs(text_encoder_output_cache_dir_name, config, tokenizer, text_encoder)
        # Move the text_encoder back to the CPU, because it is not needed for training.
        text_encoder.to("cpu")
        accelerator.wait_for_everyone()
    else:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Prepare VAE output cache.
    vae_output_cache_dir_name = None
    if config.cache_vae_outputs:
        if config.data_loader.image_transforms.random_flip:
            raise ValueError("'cache_vae_outputs' cannot be True if 'random_flip' is True.")
        if not config.data_loader.image_transforms.center_crop:
            raise ValueError("'cache_vae_outputs' cannot be True if 'center_crop' is False.")

        # We use a temporary directory for the cache. The directory will automatically be cleaned up when
        # tmp_vae_output_cache_dir is destroyed.
        tmp_vae_output_cache_dir = tempfile.TemporaryDirectory()
        vae_output_cache_dir_name = tmp_vae_output_cache_dir.name
        if accelerator.is_local_main_process:
            # Only the main process should populate the cache.
            logger.info(f"Generating VAE output cache ('{vae_output_cache_dir_name}').")
            vae.to(accelerator.device, dtype=weight_dtype)
            data_loader = build_data_loader(
                data_loader_config=config.data_loader,
                batch_size=config.train_batch_size,
                shuffle=False,
                sequential_batching=True,
            )
            cache_vae_outputs(vae_output_cache_dir_name, data_loader, vae)
        # Move the VAE back to the CPU, because it is not needed for training.
        vae.to("cpu")
        accelerator.wait_for_everyone()
    else:
        vae.to(accelerator.device, dtype=weight_dtype)

    unet.to(accelerator.device, dtype=weight_dtype)

    lora_layers = torch.nn.ModuleDict()
    trainable_param_groups = []
    if config.train_unet:
        lora_layers["unet"] = inject_lora_into_unet(
            unet, config.train_unet_non_attention_blocks, lora_rank_dim=config.lora_rank_dim
        )
        unet_param_group = {"params": lora_layers["unet"].parameters()}
        if config.unet_learning_rate is not None:
            unet_param_group["lr"] = config.unet_learning_rate
        trainable_param_groups.append(unet_param_group)
    if config.train_text_encoder:
        lora_layers["text_encoder"] = inject_lora_into_clip_text_encoder(
            text_encoder, lora_rank_dim=config.lora_rank_dim
        )
        text_encoder_param_group = {"params": lora_layers["text_encoder"].parameters()}
        if config.text_encoder_learning_rate is not None:
            text_encoder_param_group["lr"] = config.text_encoder_learning_rate
        trainable_param_groups.append(text_encoder_param_group)

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        # unet must be in train() mode for gradient checkpointing to take effect.
        # At the time of writing, the unet dropout probabilities default to 0, so putting the unet in train mode does
        # not change its forward behavior.
        unet.train()
        if config.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
            # text_encoder must be in train() mode for gradient checkpointing to take effect.
            # At the time of writing, the text_encoder dropout probabilities default to 0, so putting the text_encoder
            # in train mode does not change its forward behavior.
            text_encoder.train()

            # Set requires_grad = True on the first parameters of the text encoder. Without this, the text encoder LoRA
            # would have 0 gradients, and so would not get trained.
            text_encoder.text_model.embeddings.requires_grad_(True)

    optimizer = initialize_optimizer(config.optimizer, trainable_param_groups)

    data_loader = build_data_loader(
        data_loader_config=config.data_loader,
        batch_size=config.train_batch_size,
        text_encoder_output_cache_dir=text_encoder_output_cache_dir_name,
        vae_output_cache_dir=vae_output_cache_dir_name,
    )

    # TODO(ryand): Test in a distributed training environment and more clearly document the rationale for scaling steps
    # by the number of processes. This scaling logic was copied from the diffusers example training code, but it appears
    # in many places so I don't know where it originated. Internally, accelerate makes one LR scheduler step per process
    # (https://github.com/huggingface/accelerate/blame/49cb83a423f2946059117d8bb39b7c8747d29d80/src/accelerate/scheduler.py#L72-L82),
    # so the scaling here simply reverses that behaviour.
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = get_scheduler(
        config.optimizer.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.optimizer.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )

    prepared_result: tuple[
        UNet2DConditionModel,
        CLIPTextModel,
        torch.nn.ModuleDict,
        torch.optim.Optimizer,
        torch.utils.data.DataLoader,
        torch.optim.lr_scheduler.LRScheduler,
    ] = accelerator.prepare(
        unet,
        text_encoder,
        lora_layers,
        optimizer,
        data_loader,
        lr_scheduler,
        # Disable automatic device placement for text_encoder if the text encoder outputs were cached.
        device_placement=[True, not config.cache_text_encoder_outputs, True, True, True, True],
    )
    unet, text_encoder, lora_layers, optimizer, data_loader, lr_scheduler = prepared_result

    # Calculate the number of epochs and total training steps. A "step" represents a single weight update operation
    # (i.e. takes into account gradient accumulation steps).
    # math.ceil(...) is used in calculating the num_steps_per_epoch, because by default an optimizer step is taken when
    # the end of the dataloader is reached, even if gradient_accumulation_steps hasn't been reached.
    num_steps_per_epoch = math.ceil(len(data_loader) / config.gradient_accumulation_steps)
    num_train_epochs = math.ceil(config.max_train_steps / num_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("lora_training")
        # Tensorboard uses markdown formatting, so we wrap the config json in a code block.
        accelerator.log({"configuration": f"```json\n{json.dumps(config.dict(), indent=2, default=str)}\n```\n"})

    epoch_checkpoint_tracker = CheckpointTracker(
        base_dir=out_dir,
        prefix="checkpoint_epoch",
        extension=f".{config.output.save_model_as}",
        max_checkpoints=config.max_checkpoints,
    )

    step_checkpoint_tracker = CheckpointTracker(
        base_dir=out_dir,
        prefix="checkpoint_step",
        extension=f".{config.output.save_model_as}",
        max_checkpoints=config.max_checkpoints,
    )

    # Train!
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(data_loader)}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Parallel processes = {accelerator.num_processes}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")

    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
        range(global_step, config.max_train_steps),
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        lora_layers.train()

        train_loss = 0.0
        for data_batch in data_loader:
            with accelerator.accumulate(lora_layers):
                loss = train_forward(
                    config,
                    data_batch,
                    vae,
                    noise_scheduler,
                    tokenizer,
                    text_encoder,
                    unet,
                    weight_dtype,
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                # TODO(ryand): Test that this works properly with distributed training.
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps

                # Backpropagate.
                accelerator.backward(loss)
                if accelerator.sync_gradients and config.max_grad_norm is not None:
                    params_to_clip = lora_layers.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes.
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                log = {"train_loss": train_loss}

                lrs = lr_scheduler.get_last_lr()
                if config.train_unet:
                    # When training the UNet, it will always be the first parameter group.
                    log["lr/unet"] = float(lrs[0])
                    if config.optimizer.optimizer.optimizer_type == "Prodigy":
                        log["lr/d*lr/unet"] = optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
                if config.train_text_encoder:
                    # When training the text encoder, it will always be the last parameter group.
                    log["lr/text_encoder"] = float(lrs[-1])
                    if config.optimizer.optimizer.optimizer_type == "Prodigy":
                        log["lr/d*lr/text_encoder"] = optimizer.param_groups[-1]["d"] * optimizer.param_groups[-1]["lr"]

                accelerator.log(log, step=global_step)
                train_loss = 0.0

                if config.save_every_n_steps is not None and (global_step + 1) % config.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_lora_checkpoint(global_step + 1, lora_layers, logger, step_checkpoint_tracker)

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= config.max_train_steps:
                break

        # Save a checkpoint every n epochs.
        if config.save_every_n_epochs is not None and (epoch + 1) % config.save_every_n_epochs == 0:
            if accelerator.is_main_process:
                save_lora_checkpoint(epoch + 1, lora_layers, logger, epoch_checkpoint_tracker)
                accelerator.wait_for_everyone()

        # Generate validation images every n epochs.
        if len(config.validation_prompts) > 0 and (epoch + 1) % config.validate_every_n_epochs == 0:
            if accelerator.is_main_process:
                generate_validation_images(
                    epoch=epoch + 1,
                    out_dir=out_dir,
                    accelerator=accelerator,
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    noise_scheduler=noise_scheduler,
                    unet=unet,
                    config=config,
                    logger=logger,
                )

    accelerator.end_training()

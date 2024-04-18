import itertools
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional, Union

import peft
import torch
import torch.utils.data
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPPreTrainedModel, CLIPTextModel, PreTrainedTokenizer

from invoke_training._shared.accelerator.accelerator_utils import (
    get_mixed_precision_dtype,
    initialize_accelerator,
    initialize_logging,
)
from invoke_training._shared.checkpoints.checkpoint_tracker import CheckpointTracker
from invoke_training._shared.data.data_loaders.dreambooth_sd_dataloader import build_dreambooth_sd_dataloader
from invoke_training._shared.data.data_loaders.image_caption_sd_dataloader import build_image_caption_sd_dataloader
from invoke_training._shared.data.samplers.aspect_ratio_bucket_batch_sampler import log_aspect_ratio_buckets
from invoke_training._shared.data.transforms.tensor_disk_cache import TensorDiskCache
from invoke_training._shared.data.utils.resolution import Resolution
from invoke_training._shared.optimizer.optimizer_utils import initialize_optimizer
from invoke_training._shared.stable_diffusion.lora_checkpoint_utils import (
    TEXT_ENCODER_TARGET_MODULES,
    UNET_TARGET_MODULES,
    save_sdxl_kohya_checkpoint,
    save_sdxl_peft_checkpoint,
)
from invoke_training._shared.stable_diffusion.min_snr_weighting import compute_snr
from invoke_training._shared.stable_diffusion.model_loading_utils import load_models_sdxl
from invoke_training._shared.stable_diffusion.tokenize_captions import tokenize_captions
from invoke_training._shared.stable_diffusion.validation import generate_validation_images_sdxl
from invoke_training.config.data.data_loader_config import DreamboothSDDataLoaderConfig, ImageCaptionSDDataLoaderConfig
from invoke_training.pipelines.stable_diffusion.lora.train import cache_vae_outputs
from invoke_training.pipelines.stable_diffusion_xl.lora.config import SdxlLoraConfig


def _save_sdxl_lora_checkpoint(
    epoch: int,
    step: int,
    unet: peft.PeftModel | None,
    text_encoder_1: peft.PeftModel | None,
    text_encoder_2: peft.PeftModel | None,
    logger: logging.Logger,
    checkpoint_tracker: CheckpointTracker,
    lora_checkpoint_format: Literal["invoke_peft", "kohya"],
):
    # Prune checkpoints and get new checkpoint path.
    num_pruned = checkpoint_tracker.prune(1)
    if num_pruned > 0:
        logger.info(f"Pruned {num_pruned} checkpoint(s).")
    save_path = checkpoint_tracker.get_path(epoch=epoch, step=step)

    if lora_checkpoint_format == "invoke_peft":
        save_sdxl_peft_checkpoint(
            Path(save_path), unet=unet, text_encoder_1=text_encoder_1, text_encoder_2=text_encoder_2
        )
    elif lora_checkpoint_format == "kohya":
        save_sdxl_kohya_checkpoint(
            Path(save_path), unet=unet, text_encoder_1=text_encoder_1, text_encoder_2=text_encoder_2
        )
    else:
        raise ValueError(f"Unsupported lora_checkpoint_format: '{lora_checkpoint_format}'.")


def _build_data_loader(
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
            text_encoder_cache_field_to_output_field={
                "prompt_embeds": "prompt_embeds",
                "pooled_prompt_embeds": "pooled_prompt_embeds",
            },
            vae_output_cache_dir=vae_output_cache_dir,
            shuffle=shuffle,
        )
    elif data_loader_config.type == "DREAMBOOTH_SD_DATA_LOADER":
        return build_dreambooth_sd_dataloader(
            config=data_loader_config,
            batch_size=batch_size,
            text_encoder_output_cache_dir=text_encoder_output_cache_dir,
            text_encoder_cache_field_to_output_field={
                "prompt_embeds": "prompt_embeds",
                "pooled_prompt_embeds": "pooled_prompt_embeds",
            },
            vae_output_cache_dir=vae_output_cache_dir,
            shuffle=shuffle,
            sequential_batching=sequential_batching,
        )
    else:
        raise ValueError(f"Unsupported data loader config type: '{data_loader_config.type}'.")


# encode_prompt was adapted from:
# https://github.com/huggingface/diffusers/blob/7b07f9812a58bfa96c06ed8ffe9e6b584286e2fd/examples/text_to_image/train_text_to_image_lora_sdxl.py#L470-L496
def _encode_prompt(text_encoders: list[CLIPPreTrainedModel], prompt_token_ids_list: list[torch.Tensor]):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        text_input_ids = prompt_token_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder.
        # TODO(ryand): Document this logic more clearly.
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


# TODO(ryand): Cache VAE outputs and text encoder outputs at the same time in a single pass over the dataset.


def cache_text_encoder_outputs(
    cache_dir: str,
    config: SdxlLoraConfig,
    tokenizer_1: PreTrainedTokenizer,
    tokenizer_2: PreTrainedTokenizer,
    text_encoder_1: CLIPPreTrainedModel,
    text_encoder_2: CLIPPreTrainedModel,
):
    """Run the text encoder on all captions in the dataset and cache the results to disk.
    Args:
        cache_dir (str): The directory where the results will be cached.
        config (FinetuneLoRAConfig): Training config.
        tokenizer_1 (PreTrainedTokenizer):
        tokenizer_2 (PreTrainedTokenizer):
        text_encoder_1 (CLIPPreTrainedModel):
        text_encoder_2 (CLIPPreTrainedModel):
    """
    data_loader = _build_data_loader(
        data_loader_config=config.data_loader,
        batch_size=config.train_batch_size,
        shuffle=False,
        sequential_batching=True,
    )

    cache = TensorDiskCache(cache_dir)

    for data_batch in tqdm(data_loader):
        caption_token_ids_1 = tokenize_captions(tokenizer_1, data_batch["caption"])
        caption_token_ids_2 = tokenize_captions(tokenizer_2, data_batch["caption"])
        prompt_embeds, pooled_prompt_embeds = _encode_prompt(
            [text_encoder_1, text_encoder_2], [caption_token_ids_1, caption_token_ids_2]
        )

        # Split batch before caching.
        for i in range(len(data_batch["id"])):
            embeds = {
                "prompt_embeds": prompt_embeds[i],
                "pooled_prompt_embeds": pooled_prompt_embeds[i],
            }
            cache.save(data_batch["id"][i], embeds)


def train_forward(  # noqa: C901
    accelerator: Accelerator,
    data_batch: dict,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    tokenizer_1: PreTrainedTokenizer,
    tokenizer_2: PreTrainedTokenizer,
    text_encoder_1: CLIPPreTrainedModel,
    text_encoder_2: CLIPPreTrainedModel,
    unet: UNet2DConditionModel,
    weight_dtype: torch.dtype,
    resolution: int | tuple[int, int],
    prediction_type=None,
    min_snr_gamma: float | None = None,
):
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

    # Add noise to the latents according to the noise magnitude at each timestep (this is the forward diffusion
    # process).
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # compute_time_ids was copied from:
    # https://github.com/huggingface/diffusers/blob/7b07f9812a58bfa96c06ed8ffe9e6b584286e2fd/examples/text_to_image/train_text_to_image_lora_sdxl.py#L1033-L1039
    # "time_ids" may seem like a weird naming choice. The name comes from the diffusers SDXL implementation. Presumably,
    # it is a result of the fact that the original size and crop values get concatenated with the time embeddings.
    def compute_time_ids(original_size, crops_coords_top_left):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = Resolution.parse(resolution).to_tuple()
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    add_time_ids = torch.cat(
        [compute_time_ids(s, c) for s, c in zip(data_batch["original_size_hw"], data_batch["crop_top_left_yx"])]
    )
    unet_conditions = {"time_ids": add_time_ids}

    # Get the text embedding for conditioning.
    # The text encoder output may have been cached and included in the data_batch. If not, we calculate it here.
    if "prompt_embeds" in data_batch:
        prompt_embeds = data_batch["prompt_embeds"]
        pooled_prompt_embeds = data_batch["pooled_prompt_embeds"]
    else:
        caption_token_ids_1 = tokenize_captions(tokenizer_1, data_batch["caption"])
        caption_token_ids_2 = tokenize_captions(tokenizer_2, data_batch["caption"])
        prompt_embeds, pooled_prompt_embeds = _encode_prompt(
            [text_encoder_1, text_encoder_2], [caption_token_ids_1, caption_token_ids_2]
        )
        prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)

    unet_conditions["text_embeds"] = pooled_prompt_embeds

    # Get the target for loss depending on the prediction type.
    if prediction_type is not None:
        # Set the prediction_type of scheduler if it's defined in config.
        noise_scheduler.register_to_config(prediction_type=prediction_type)
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    # Predict the noise residual.
    model_pred = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=unet_conditions).sample

    min_snr_weights = None
    if min_snr_gamma is not None:
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.

        snr = compute_snr(noise_scheduler, timesteps)

        # Note: We divide by snr here per Section 4.2 of the paper, since we are predicting the noise instead of x_0.
        # w_t = min(1, SNR(t)) / SNR(t)
        min_snr_weights = torch.clamp(snr, max=min_snr_gamma) / snr

        if noise_scheduler.config.prediction_type == "epsilon":
            pass
        elif noise_scheduler.config.prediction_type == "v_prediction":
            # Velocity objective needs to be floored to an SNR weight of one.
            min_snr_weights = min_snr_weights + 1
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")

    # Mean-reduce the loss along all dimensions except for the batch dimension.
    loss = loss.mean(dim=list(range(1, len(loss.shape))))

    # Apply min_snr_weights.
    if min_snr_weights is not None:
        loss = loss * min_snr_weights

    # Apply per-example loss weights.
    if "loss_weight" in data_batch:
        loss = loss * data_batch["loss_weight"]

    return loss.mean()


def train(config: SdxlLoraConfig):  # noqa: C901
    # Give a clear error message if an unsupported base model was chosen.
    # TODO(ryan): Update this check to work with single-file SD checkpoints.
    # check_base_model_version(
    #     {BaseModelVersionEnum.STABLE_DIFFUSION_SDXL_BASE},
    #     config.model,
    #     local_files_only=False,
    # )

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

    logger.info("Starting Training.")
    logger.info(f"Configuration:\n{json.dumps(config.dict(), indent=2, default=str)}")
    logger.info(f"Output dir: '{out_dir}'")

    # Write the configuration to disk.
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config.dict(), f, indent=2, default=str)

    weight_dtype = get_mixed_precision_dtype(accelerator)

    logger.info("Loading models.")
    tokenizer_1, tokenizer_2, noise_scheduler, text_encoder_1, text_encoder_2, vae, unet = load_models_sdxl(
        model_name_or_path=config.model,
        hf_variant=config.hf_variant,
        vae_model=config.vae_model,
        base_embeddings=config.base_embeddings,
    )

    if config.xformers:
        import xformers  # noqa: F401

        # TODO(ryand): There is a known issue if xformers is enabled when training in mixed precision where xformers
        # will fail because Q, K, V have different dtypes.
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
            text_encoder_1.to(accelerator.device, dtype=weight_dtype)
            text_encoder_2.to(accelerator.device, dtype=weight_dtype)
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
    vae_output_cache_dir_name = None
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
            # Only the main process should to populate the cache.
            logger.info(f"Generating VAE output cache ('{vae_output_cache_dir_name}').")
            vae.to(accelerator.device, dtype=weight_dtype)
            data_loader = _build_data_loader(
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

    if config.train_unet:
        unet_lora_config = peft.LoraConfig(
            r=config.lora_rank_dim,
            # TODO(ryand): Diffusers uses lora_alpha=config.lora_rank_dim. Is that preferred?
            lora_alpha=1.0,
            target_modules=UNET_TARGET_MODULES,
        )
        unet = inject_lora_layers(unet, unet_lora_config, lr=config.unet_learning_rate)

    if config.train_text_encoder:
        text_encoder_lora_config = peft.LoraConfig(
            r=config.lora_rank_dim,
            lora_alpha=1.0,
            # init_lora_weights="gaussian",
            target_modules=TEXT_ENCODER_TARGET_MODULES,
        )
        text_encoder_1 = inject_lora_layers(
            text_encoder_1, text_encoder_lora_config, lr=config.text_encoder_learning_rate
        )
        text_encoder_2 = inject_lora_layers(
            text_encoder_2, text_encoder_lora_config, lr=config.text_encoder_learning_rate
        )

    # Make sure all trainable params are in float32.
    for trainable_model in all_trainable_models:
        for param in trainable_model.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)

    if config.gradient_checkpointing:
        # We want to enable gradient checkpointing in the UNet regardless of whether it is being trained.
        unet.enable_gradient_checkpointing()
        # unet must be in train() mode for gradient checkpointing to take effect.
        # At the time of writing, the unet dropout probabilities default to 0, so putting the unet in train mode does
        # not change its forward behavior.
        unet.train()
        if config.train_text_encoder:
            for te in [text_encoder_1, text_encoder_2]:
                te.gradient_checkpointing_enable()

                # The text encoders must be in train() mode for gradient checkpointing to take effect. This should
                # already be the case, since we are training the text_encoders, be we do it explicitly to make it clear
                # that this is required.
                # At the time of writing, the text encoder dropout probabilities default to 0, so putting the text
                # encoders in train mode does not change their forward behavior.
                te.train()

                # Set requires_grad = True on the first parameters of the text encoders. Without this, the text encoder
                # LoRA weights would have 0 gradients, and so would not get trained. Note that the set of
                # trainable_param_groups has already been populated - the embeddings will not be trained.
                te.text_model.embeddings.requires_grad_(True)

    optimizer = initialize_optimizer(config.optimizer, trainable_param_groups)

    data_loader = _build_data_loader(
        data_loader_config=config.data_loader,
        batch_size=config.train_batch_size,
        text_encoder_output_cache_dir=text_encoder_output_cache_dir_name,
        vae_output_cache_dir=vae_output_cache_dir_name,
    )

    log_aspect_ratio_buckets(logger=logger, batch_sampler=data_loader.batch_sampler)

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
        UNet2DConditionModel,
        peft.PeftModel | CLIPTextModel,
        peft.PeftModel | CLIPTextModel,
        torch.optim.Optimizer,
        torch.utils.data.DataLoader,
        torch.optim.lr_scheduler.LRScheduler,
    ] = accelerator.prepare(
        unet,
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
    unet, text_encoder_1, text_encoder_2, optimizer, data_loader, lr_scheduler = prepared_result

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
            _save_sdxl_lora_checkpoint(
                epoch=num_completed_epochs,
                step=num_completed_steps,
                unet=unet if config.train_unet else None,
                text_encoder_1=text_encoder_1 if config.train_text_encoder else None,
                text_encoder_2=text_encoder_2 if config.train_text_encoder else None,
                logger=logger,
                checkpoint_tracker=checkpoint_tracker,
                lora_checkpoint_format=config.lora_checkpoint_format,
            )
        accelerator.wait_for_everyone()

    def validate(num_completed_epochs: int, num_completed_steps: int):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            generate_validation_images_sdxl(
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
                unet=unet,
                config=config,
                logger=logger,
            )
        accelerator.wait_for_everyone()

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for data_batch_idx, data_batch in enumerate(data_loader):
            with accelerator.accumulate(unet, text_encoder_1, text_encoder_2):
                loss = train_forward(
                    accelerator=accelerator,
                    data_batch=data_batch,
                    vae=vae,
                    noise_scheduler=noise_scheduler,
                    tokenizer_1=tokenizer_1,
                    tokenizer_2=tokenizer_2,
                    text_encoder_1=text_encoder_1,
                    text_encoder_2=text_encoder_2,
                    unet=unet,
                    weight_dtype=weight_dtype,
                    resolution=config.data_loader.resolution,
                    prediction_type=config.prediction_type,
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
                if config.train_unet:
                    # When training the UNet, it will always be the first parameter group.
                    log["lr/unet"] = float(lrs[0])
                    if config.optimizer.optimizer_type == "Prodigy":
                        log["lr/d*lr/unet"] = optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
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

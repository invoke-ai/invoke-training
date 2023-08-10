import json
import logging
import math
import os
import time

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    CLIPPreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizer,
)

from invoke_training.lora.injection.stable_diffusion_v1 import (
    convert_lora_state_dict_to_kohya_format_sd1,
    inject_lora_into_clip_text_encoder,
    inject_lora_into_unet_sd1,
)
from invoke_training.training.finetune_lora.finetune_lora_config import (
    FinetuneLoRASDXLConfig,
)
from invoke_training.training.shared.accelerator_utils import (
    get_mixed_precision_dtype,
    initialize_accelerator,
    initialize_logging,
)
from invoke_training.training.shared.base_model_version import (
    BaseModelVersionEnum,
    check_base_model_version,
)
from invoke_training.training.shared.checkpoint_tracker import CheckpointTracker
from invoke_training.training.shared.datasets.image_caption_sdxl_dataloader import (
    build_image_caption_sdxl_dataloader,
)
from invoke_training.training.shared.serialization import save_state_dict


def _import_model_class_for_model(pretrained_model_name_or_path: str, subfolder: str = "", revision: str = "main"):
    """Lookup the model class in a diffusers model config, import the class, and return it. This function is useful when
    loading models that could be one of many possible classes.

    Args:
        pretrained_model_name_or_path (str): The diffusers model name/path.
        subfolder (str, optional): The model subfolder.
        revision (str, optional): The diffusers model revision.


    Raises:
        ValueError: If the detected model class is not recognize.

    Returns:
        type: The model class.
    """
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def _load_models(
    accelerator: Accelerator,
    config: FinetuneLoRASDXLConfig,
) -> tuple[
    PreTrainedTokenizer,
    PreTrainedTokenizer,
    DDPMScheduler,
    CLIPPreTrainedModel,
    CLIPPreTrainedModel,
    AutoencoderKL,
    UNet2DConditionModel,
]:
    """Load all models required for training, transfer them to the target training device and cast their weight dtypes.

    Args:
        accelerator (Accelerator): Accelerator
        config (FinetuneLoRASDXLConfig): Training config.

    Returns:
        tuple[
            PreTrainedTokenizer,
            PreTrainedTokenizer,
            DDPMScheduler,
            CLIPPreTrainedModel,
            CLIPPreTrainedModel,
            AutoencoderKL,
            UNet2DConditionModel,
        ]: A tuple of loaded models.
    """
    weight_dtype = get_mixed_precision_dtype(accelerator)

    # Load tokenizers.
    tokenizer_1: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        config.model, subfolder="tokenizer", use_fast=False
    )
    tokenizer_2: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        config.model, subfolder="tokenizer_2", use_fast=False
    )

    # Load noise scheduler.
    noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(config.model, subfolder="scheduler")

    # Load text encoders.
    text_encoder_cls_1 = _import_model_class_for_model(config.model, subfolder="text_encoder")
    text_encoder_1 = text_encoder_cls_1.from_pretrained(config.model, subfolder="text_encoder")
    text_encoder_cls_2 = _import_model_class_for_model(config.model, subfolder="text_encoder_2")
    text_encoder_2 = text_encoder_cls_2.from_pretrained(config.model, subfolder="text_encoder_2")

    # Load VAE.
    vae_model = config.vae_model if config.vae_model is not None else config.model
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(vae_model, subfolder="vae")

    # Load UNet.
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(config.model, subfolder="unet")

    # Disable gradient calculation for model weights to save memory.
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Put models in 'eval' mode.
    text_encoder_1.eval()
    text_encoder_2.eval()
    vae.eval()
    unet.eval()

    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    return tokenizer_1, tokenizer_2, noise_scheduler, text_encoder_1, text_encoder_2, vae, unet


# TODO(ryand): Split this function out to avoid duplication between sd and sdxl configs.
def _initialize_optimizer(config: FinetuneLoRASDXLConfig, trainable_params: list) -> torch.optim.Optimizer:
    """Initialize an optimizer based on the config."""
    return torch.optim.AdamW(
        trainable_params,
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
        weight_decay=config.optimizer.adam_weight_decay,
        eps=config.optimizer.adam_epsilon,
    )


def _save_checkpoint(
    idx: int,
    lora_layers: torch.nn.ModuleDict,
    logger: logging.Logger,
    checkpoint_tracker: CheckpointTracker,
):
    """Save a checkpoint. Old checkpoints are deleted if necessary to respect the config.max_checkpoints config.

    Args:
        idx (int): The checkpoint index (typically step count or epoch).
        lora_layers (torch.nn.ModuleDict): The LoRA layers to save in a ModuleDict mapping keys to
            `LoRALayerCollection`s.
        logger (logging.Logger): Logger.
        checkpoint_tracker (CheckpointTracker): The checkpoint tracker.
    """
    # Prune checkpoints and get new checkpoint path.
    num_pruned = checkpoint_tracker.prune(1)
    if num_pruned > 0:
        logger.info(f"Pruned {num_pruned} checkpoint(s).")
    save_path = checkpoint_tracker.get_path(idx)

    state_dict = {}
    for model_lora_layers in lora_layers.values():
        model_state_dict = model_lora_layers.get_lora_state_dict()
        model_kohya_state_dict = convert_lora_state_dict_to_kohya_format_sd1(model_state_dict)
        state_dict.update(model_kohya_state_dict)

    save_state_dict(state_dict, save_path)
    # accelerator.save_state(save_path)
    logger.info(f"Saved state to '{save_path}'.")


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


def _generate_validation_images(
    epoch: int,
    out_dir: str,
    accelerator: Accelerator,
    vae: AutoencoderKL,
    text_encoder_1: CLIPPreTrainedModel,
    text_encoder_2: CLIPPreTrainedModel,
    tokenizer_1: PreTrainedTokenizer,
    tokenizer_2: PreTrainedTokenizer,
    noise_scheduler: DDPMScheduler,
    unet: UNet2DConditionModel,
    config: FinetuneLoRASDXLConfig,
    logger: logging.Logger,
):
    """Generate validation images for the purpose of tracking image generation behaviour on fixed prompts throughout
    training.
    """
    logger.info("Generating validation images.")

    # Create pipeline.
    pipeline = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=noise_scheduler,
    )
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


def _train_forward(
    accelerator: Accelerator,
    config: FinetuneLoRASDXLConfig,
    data_batch: dict,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    text_encoder_1: CLIPPreTrainedModel,
    text_encoder_2: CLIPPreTrainedModel,
    unet: UNet2DConditionModel,
    weight_dtype: torch.dtype,
):
    """Run the forward training pass for a single data_batch.

    Returns:
        torch.Tensor: Loss
    """
    # Convert images to latent space.
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
        target_size = (config.dataset.resolution, config.dataset.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    add_time_ids = torch.cat(
        [compute_time_ids(s, c) for s, c in zip(data_batch["original_size_hw"], data_batch["crop_top_left_yx"])]
    )
    unet_conditions = {"time_ids": add_time_ids}

    # Get the text embedding for conditioning.
    prompt_embeds, pooled_prompt_embeds = _encode_prompt(
        text_encoders=[text_encoder_1, text_encoder_2],
        prompt_token_ids_list=[data_batch["caption_token_ids_1"], data_batch["caption_token_ids_2"]],
    )
    unet_conditions["text_embeds"] = pooled_prompt_embeds

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
    model_pred = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=unet_conditions).sample

    return torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")


def run_training(config: FinetuneLoRASDXLConfig):  # noqa: C901
    # Give a clear error message if an unsupported base model was chosen.
    check_base_model_version(
        {BaseModelVersionEnum.STABLE_DIFFUSION_SDXL_BASE},
        config.model,
        local_files_only=False,
    )

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

    logger.info("Starting Training.")
    logger.info(f"Configuration:\n{json.dumps(config.dict(), indent=2, default=str)}")
    logger.info(f"Output dir: '{out_dir}'")

    # Write the configuration to disk.
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config.dict(), f, indent=2, default=str)

    weight_dtype = get_mixed_precision_dtype(accelerator)

    logger.info("Loading models.")
    tokenizer_1, tokenizer_2, noise_scheduler, text_encoder_1, text_encoder_2, vae, unet = _load_models(
        accelerator, config
    )

    lora_layers = torch.nn.ModuleDict()
    if config.train_unet:
        lora_layers["unet"] = inject_lora_into_unet_sd1(unet, config.train_unet_non_attention_blocks)
    if config.train_text_encoder:
        lora_layers["text_encoder_1"] = inject_lora_into_clip_text_encoder(text_encoder_1, "lora_te1")
        lora_layers["text_encoder_2"] = inject_lora_into_clip_text_encoder(text_encoder_1, "lora_te2")

    if config.xformers:
        import xformers  # noqa: F401

        unet.enable_xformers_memory_efficient_attention()
        vae.enable_xformers_memory_efficient_attention()

    optimizer = _initialize_optimizer(config, lora_layers.parameters())

    data_loader = build_image_caption_sdxl_dataloader(config.dataset, tokenizer_1, tokenizer_2, config.train_batch_size)

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
        CLIPPreTrainedModel,
        CLIPPreTrainedModel,
        torch.nn.ModuleDict,
        torch.optim.Optimizer,
        torch.utils.data.DataLoader,
        torch.optim.lr_scheduler.LRScheduler,
    ] = accelerator.prepare(unet, text_encoder_1, text_encoder_2, lora_layers, optimizer, data_loader, lr_scheduler)
    unet, text_encoder_1, text_encoder_2, lora_layers, optimizer, data_loader, lr_scheduler = prepared_result

    # Calculate the number of epochs and total training steps. A "step" represents a single weight update operation
    # (i.e. takes into account gradient accumulation steps).
    # math.ceil(...) is used in calculating the num_steps_per_epoch, because by default an optimizer step is taken when
    # the end of the dataloader is reached, even if gradient_accumulation_steps hasn't been reached.
    num_steps_per_epoch = math.ceil(len(data_loader) / config.gradient_accumulation_steps)
    num_train_epochs = math.ceil(config.max_train_steps / num_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("lora_training")

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
        for i, data_batch in enumerate(data_loader):
            if (i + 1) % 10 == 0:
                break
            with accelerator.accumulate(lora_layers):
                loss = _train_forward(
                    accelerator,
                    config,
                    data_batch,
                    vae,
                    noise_scheduler,
                    text_encoder_1,
                    text_encoder_2,
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
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if config.save_every_n_steps is not None and (global_step + 1) % config.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        _save_checkpoint(global_step + 1, lora_layers, logger, step_checkpoint_tracker)

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
                _save_checkpoint(epoch + 1, lora_layers, logger, epoch_checkpoint_tracker)
                accelerator.wait_for_everyone()

        # Generate validation images every n epochs.
        if len(config.validation_prompts) > 0 and (epoch + 1) % config.validate_every_n_epochs == 0:
            if accelerator.is_main_process:
                _generate_validation_images(
                    epoch=epoch + 1,
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

    accelerator.end_training()

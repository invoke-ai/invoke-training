import json
import logging
import math
import os
import time

import datasets
import diffusers
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter, get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from invoke_training.lora.injection.lora_layer_collection import LoRALayerCollection
from invoke_training.lora.injection.stable_diffusion_v1 import (
    convert_lora_state_dict_to_kohya_format_sd1,
    inject_lora_into_unet_sd1,
)
from invoke_training.training.lora.lora_training_config import LoRATrainingConfig
from invoke_training.training.shared.checkpoint_tracker import CheckpointTracker
from invoke_training.training.shared.hf_dataset import initialize_hf_dataloader
from invoke_training.training.shared.serialization import save_state_dict


def _initialize_accelerator(out_dir: str, config: LoRATrainingConfig) -> Accelerator:
    """Configure Hugging Face accelerate and return an Accelerator.

    Args:
        out_dir (str): The output directory where results will be written.
        config (LoRATrainingConfig): LoRA training configuration.

    Returns:
        Accelerator
    """
    accelerator_project_config = ProjectConfiguration(
        project_dir=out_dir,
        logging_dir=os.path.join(out_dir, "logs"),
    )
    return Accelerator(
        project_config=accelerator_project_config,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.output.report_to,
    )


def _initialize_logging(accelerator: Accelerator) -> MultiProcessAdapter:
    """Configure logging.

    Returns an accelerate logger with multi-process logging support. Logging is configured to be more verbose on the
    main process. Non-main processes only log at error level for Hugging Face libraries (datasets, transformers,
    diffusers).

    Args:
        accelerator (Accelerator): The Accelerator to configure.

    Returns:
        MultiProcessAdapter: _description_
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # Only log errors from non-main processes.
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    return get_logger(__name__)


def _get_weight_type(accelerator: Accelerator):
    """Extract torch.dtype from Accelerator config.

    Args:
        accelerator (Accelerator): The Hugging Face Accelerator.

    Raises:
        NotImplementedError: If the accelerator's mixed_precision configuration is not recognized.

    Returns:
        torch.dtype: The weight type inferred from the accelerator mixed_precision configuration.
    """
    weight_dtype: torch.dtype = torch.float32
    if accelerator.mixed_precision is None or accelerator.mixed_precision == "no":
        weight_dtype = torch.float32
    elif accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        raise NotImplementedError(f"mixed_precision mode '{accelerator.mixed_precision}' is not yet supported.")
    return weight_dtype


def _load_models(
    accelerator: Accelerator,
    config: LoRATrainingConfig,
) -> tuple[CLIPTokenizer, DDPMScheduler, CLIPTextModel, AutoencoderKL, UNet2DConditionModel]:
    """Load all models required for training from disk, transfer them to the
    target training device and cast their weight dtypes.

    Args:
        config (LoRATrainingConfig): The LoRA training run config.
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
    weight_dtype = _get_weight_type(accelerator)

    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(config.model, subfolder="tokenizer", local_files_only=True)
    noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
        config.model, subfolder="scheduler", local_files_only=True
    )
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
        config.model,
        subfolder="text_encoder",
        local_files_only=True,
    )
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(config.model, subfolder="vae", local_files_only=True)
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        config.model,
        subfolder="unet",
        local_files_only=True,
    )

    # Disable gradient calculation for model weights to save memory.
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Put models in 'eval' mode.
    text_encoder.eval()
    vae.eval()
    unet.eval()

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    return tokenizer, noise_scheduler, text_encoder, vae, unet


def _initialize_optimizer(config: LoRATrainingConfig, trainable_params: list) -> torch.optim.Optimizer:
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
    lora_layers: LoRALayerCollection,
    logger: logging.Logger,
    checkpoint_tracker: CheckpointTracker,
):
    """Save a checkpoint. Old checkpoints are deleted if necessary to respect the config.max_checkpoints config.

    Args:
        idx (int): The checkpoint index (typically step count or epoch).
        lora_layers (LoRALayerCollection): The LoRA layers to save.
        logger (logging.Logger): Logger.
        checkpoint_tracker (CheckpointTracker): The checkpoint tracker.
    """
    num_pruned = checkpoint_tracker.prune(1)
    if num_pruned > 0:
        logger.info(f"Pruned {num_pruned} checkpoint(s).")
    save_path = checkpoint_tracker.get_path(idx)

    state_dict = lora_layers.get_lora_state_dict()
    kohya_state_dict = convert_lora_state_dict_to_kohya_format_sd1(state_dict)
    save_state_dict(kohya_state_dict, save_path)
    # accelerator.save_state(save_path)
    logger.info(f"Saved state to '{save_path}'.")


def _generate_validation_images(
    epoch: int,
    out_dir: str,
    accelerator: Accelerator,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    noise_scheduler: DDPMScheduler,
    unet: UNet2DConditionModel,
    config: LoRATrainingConfig,
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
        config (LoRATrainingConfig): Training configs.
        logger (logging.Logger): Logger.
    """
    logger.info("Generating validation images.")

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
    config: LoRATrainingConfig,
    data_batch: dict,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    text_encoder: CLIPTextModel,
    unet: UNet2DConditionModel,
    weight_dtype: torch.dtype,
):
    """Run the forward training pass for a single data_batch.

    Returns:
        torch.Tensor: Loss
    """
    # Convert images to latent space.
    latents = vae.encode(data_batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
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
    encoder_hidden_states = text_encoder(data_batch["input_ids"])[0]

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

    return torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")


def run_lora_training(config: LoRATrainingConfig):  # noqa: C901
    # Create a timestamped directory for all outputs.
    out_dir = os.path.join(config.output.base_output_dir, f"{time.time()}")
    os.makedirs(out_dir)

    accelerator = _initialize_accelerator(out_dir, config)
    logger = _initialize_logging(accelerator)

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

    weight_dtype = _get_weight_type(accelerator)

    logger.info("Loading models.")
    tokenizer, noise_scheduler, text_encoder, vae, unet = _load_models(accelerator, config)

    unet_lora_layers = inject_lora_into_unet_sd1(unet)

    if config.xformers:
        import xformers  # noqa: F401

        unet.enable_xformers_memory_efficient_attention()
        vae.enable_xformers_memory_efficient_attention()

    optimizer = _initialize_optimizer(config, unet_lora_layers.parameters())

    data_loader = initialize_hf_dataloader(config.dataset, accelerator, tokenizer, config.train_batch_size)

    # TODO(ryand): Revisit and more clearly document the definition of 'steps'. Consider interactions with batch_size,
    # gradient_accumulation_steps, and number of training processes.
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = get_scheduler(
        config.optimizer.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.optimizer.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )

    prepared_result: tuple[
        UNet2DConditionModel,
        CLIPTextModel,
        LoRALayerCollection,
        torch.optim.Optimizer,
        torch.utils.data.DataLoader,
        torch.optim.lr_scheduler.LRScheduler,
    ] = accelerator.prepare(unet, text_encoder, unet_lora_layers, optimizer, data_loader, lr_scheduler)
    unet, text_encoder, unet_lora_layers, optimizer, data_loader, lr_scheduler = prepared_result

    # Calculate number of epochs and total training steps.
    # Note: A "step" represents a single optimizer weight update operation (i.e. takes into account gradient
    # accumulation steps).
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
        unet_lora_layers.train()

        train_loss = 0.0
        for data_batch in data_loader:
            with accelerator.accumulate(unet_lora_layers):
                loss = _train_forward(
                    config,
                    data_batch,
                    vae,
                    noise_scheduler,
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
                    params_to_clip = unet_lora_layers.parameters()
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
                        _save_checkpoint(global_step + 1, unet_lora_layers, logger, step_checkpoint_tracker)

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
                _save_checkpoint(epoch + 1, unet_lora_layers, logger, epoch_checkpoint_tracker)
                accelerator.wait_for_everyone()

        # Generate validation images every n epochs.
        if len(config.validation_prompts) > 0 and (epoch + 1) % config.validate_every_n_epochs == 0:
            if accelerator.is_main_process:
                _generate_validation_images(
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

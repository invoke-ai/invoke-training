import json
import math
import os
import time

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPPreTrainedModel, PreTrainedTokenizer

from invoke_training.lora.injection.stable_diffusion import (
    inject_lora_into_clip_text_encoder,
    inject_lora_into_unet,
)
from invoke_training.training.config.finetune_lora_config import (
    DreamBoothLoRASDXLConfig,
)
from invoke_training.training.finetune_lora.finetune_lora_sdxl import (
    encode_prompt,
    generate_validation_images,
    inference_pipeline,
    load_models,
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
from invoke_training.training.shared.data.data_loaders.dreambooth_sdxl_dataloader import (
    build_dreambooth_sdxl_dataloader,
)
from invoke_training.training.shared.data.datasets.image_dir_dataset import (
    ImageDirDataset,
)
from invoke_training.training.shared.lora_checkpoint_utils import save_lora_checkpoint
from invoke_training.training.shared.optimizer_utils import initialize_optimizer


def _generate_class_images(
    config: DreamBoothLoRASDXLConfig,
    accelerator: Accelerator,
    vae: AutoencoderKL,
    text_encoder_1: CLIPPreTrainedModel,
    text_encoder_2: CLIPPreTrainedModel,
    tokenizer_1: PreTrainedTokenizer,
    tokenizer_2: PreTrainedTokenizer,
    noise_scheduler: DDPMScheduler,
    unet: UNet2DConditionModel,
):
    """Generate a prior-preservation class image dataset based on config."""
    os.makedirs(config.class_image_dir, exist_ok=True)
    with inference_pipeline(
        accelerator,
        vae,
        text_encoder_1,
        text_encoder_2,
        tokenizer_1,
        tokenizer_2,
        noise_scheduler,
        unet,
        config.enable_cpu_offload_during_validation,
    ) as pipeline, torch.no_grad():
        generator = torch.Generator(device=accelerator.device)
        if config.seed is not None:
            generator = generator.manual_seed(config.seed)
        for image_idx in tqdm(range(config.num_class_images), disable=not accelerator.is_local_main_process):
            for _ in range(config.num_validation_images_per_prompt):
                with accelerator.autocast():
                    image = pipeline(
                        config.class_prompt,
                        num_inference_steps=30,
                        generator=generator,
                        height=config.instance_dataset.image_transforms.resolution,
                        width=config.instance_dataset.image_transforms.resolution,
                    ).images[0]

                    image.save(os.path.join(config.class_image_dir, f"{image_idx:0>4}.png"))


def _train_forward(
    accelerator: Accelerator,
    config: DreamBoothLoRASDXLConfig,
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
        target_size = (
            config.instance_dataset.image_transforms.resolution,
            config.instance_dataset.image_transforms.resolution,
        )
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    add_time_ids = torch.cat(
        [compute_time_ids(s, c) for s, c in zip(data_batch["original_size_hw"], data_batch["crop_top_left_yx"])]
    )
    unet_conditions = {"time_ids": add_time_ids}

    # Get the text embedding for conditioning.
    # The text_encoder_output may have been cached and included in the data_batch. If not, we calculate it here.
    all_prompt_embeds = data_batch.get("text_encoder_output", None)
    if all_prompt_embeds is None:
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders=[text_encoder_1, text_encoder_2],
            prompt_token_ids_list=[data_batch["caption_token_ids_1"], data_batch["caption_token_ids_2"]],
        )
    else:
        prompt_embeds = all_prompt_embeds["prompt_embeds"]
        pooled_prompt_embeds = all_prompt_embeds["pooled_prompt_embeds"]

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

    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
    # Mean-reduce the loss along all dimensions except for the batch dimension.
    loss = loss.mean([1, 2, 3])
    # Apply per-example weights.
    loss = loss * data_batch["loss_weight"]
    loss = loss.mean()
    return loss


def run_training(config: DreamBoothLoRASDXLConfig):  # noqa: C901
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
    tokenizer_1, tokenizer_2, noise_scheduler, text_encoder_1, text_encoder_2, vae, unet = load_models(config)

    if config.xformers:
        import xformers  # noqa: F401

        unet.enable_xformers_memory_efficient_attention()
        vae.enable_xformers_memory_efficient_attention()

    # Generate class images if prior preservation is enabled.
    if config.use_prior_preservation and accelerator.is_main_process:
        class_image_dataset = None
        if os.path.exists(config.class_image_dir):
            class_image_dataset = ImageDirDataset(config.class_image_dir)

        if class_image_dataset is not None and len(class_image_dataset) > 0:
            logger.info(
                f"Detected an existing dataset at '{config.class_image_dir}' with {len(class_image_dataset)} images. "
                "Using this class image dataset for prior-preservation."
            )
        else:
            logger.info(
                f"No existing dataset found at '{config.class_image_dir}'. "
                f"Generating {config.num_class_images} prior-preservation class images."
            )
            _generate_class_images(
                config,
                accelerator,
                vae,
                text_encoder_1,
                text_encoder_2,
                tokenizer_1,
                tokenizer_2,
                noise_scheduler,
                unet,
            )
    accelerator.wait_for_everyone()

    # Prepare text encoder output cache.
    # text_encoder_output_cache_dir_name = None
    if config.cache_text_encoder_outputs:
        if config.train_text_encoder:
            raise ValueError("'cache_text_encoder_outputs' and 'train_text_encoder' cannot both be True.")
        raise NotImplementedError("'cache_text_encoder_outputs' is not yet supported in DreamBooth training.")
    else:
        text_encoder_1.to(accelerator.device, dtype=weight_dtype)
        text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # Prepare VAE output cache.
    # vae_output_cache_dir_name = None
    if config.cache_vae_outputs:
        if config.dataset.image_transforms.random_flip:
            raise ValueError("'cache_vae_outputs' cannot be True if 'random_flip' is True.")
        if not config.dataset.image_transforms.center_crop:
            raise ValueError("'cache_vae_outputs' cannot be True if 'center_crop' is False.")

        raise NotImplementedError("'cache_vae_outputs' is not yet supported in DreamBooth training.")
    else:
        vae.to(accelerator.device, dtype=weight_dtype)

    unet.to(accelerator.device, dtype=weight_dtype)

    lora_layers = torch.nn.ModuleDict()
    if config.train_unet:
        lora_layers["unet"] = inject_lora_into_unet(
            unet, config.train_unet_non_attention_blocks, lora_rank_dim=config.lora_rank_dim
        )
    if config.train_text_encoder:
        lora_layers["text_encoder_1"] = inject_lora_into_clip_text_encoder(
            text_encoder_1, "lora_te1", lora_rank_dim=config.lora_rank_dim
        )
        lora_layers["text_encoder_2"] = inject_lora_into_clip_text_encoder(
            text_encoder_2, "lora_te2", lora_rank_dim=config.lora_rank_dim
        )

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        # unet must be in train() mode for gradient checkpointing to take effect.
        # At the time of writing, the unet dropout probabilities default to 0, so putting the unet in train mode does
        # not change its forward behavior.
        unet.train()
        if config.train_text_encoder:
            for te in [text_encoder_1, text_encoder_2]:
                te.gradient_checkpointing_enable()

                # The text encoders must be in train() mode for gradient checkpointing to take effect.
                # At the time of writing, the text encoder dropout probabilities default to 0, so putting the text
                # encoders in train mode does not change their forward behavior.
                te.train()

                # Set requires_grad = True on the first parameters of the text encoders. Without this, the text encoder
                # LoRA weights would have 0 gradients, and so would not get trained.
                te.text_model.embeddings.requires_grad_(True)

    optimizer = initialize_optimizer(config.optimizer, lora_layers.parameters())

    data_loader = build_dreambooth_sdxl_dataloader(
        instance_prompt=config.instance_prompt,
        instance_dataset_config=config.instance_dataset,
        class_prompt=config.class_prompt,
        class_data_dir=config.class_image_dir if config.use_prior_preservation else None,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        batch_size=config.train_batch_size,
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
        CLIPPreTrainedModel,
        CLIPPreTrainedModel,
        torch.nn.ModuleDict,
        torch.optim.Optimizer,
        torch.utils.data.DataLoader,
        torch.optim.lr_scheduler.LRScheduler,
    ] = accelerator.prepare(
        unet,
        text_encoder_1,
        text_encoder_2,
        lora_layers,
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
            True,
        ],
    )
    unet, text_encoder_1, text_encoder_2, lora_layers, optimizer, data_loader, lr_scheduler = prepared_result

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
                log = {"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}
                if config.optimizer.optimizer.optimizer_type == "Prodigy":
                    log["lr/d*lr"] = optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
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

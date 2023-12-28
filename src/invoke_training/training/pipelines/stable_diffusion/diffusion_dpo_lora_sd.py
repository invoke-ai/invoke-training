import copy
import json
import math
import os
import tempfile
import time

import torch
import torch.utils.data
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from invoke_training.config.pipelines.finetune_lora_config import (
    DirectPreferenceOptimizationLoRASDConfig,
)
from invoke_training.core.lora.injection.stable_diffusion import (
    inject_lora_into_clip_text_encoder,
    inject_lora_into_unet,
)
from invoke_training.training.pipelines.stable_diffusion.finetune_lora_sd import (
    cache_text_encoder_outputs,
    generate_validation_images,
    load_models,
)
from invoke_training.training.shared.accelerator.accelerator_utils import (
    get_mixed_precision_dtype,
    initialize_accelerator,
    initialize_logging,
)
from invoke_training.training.shared.checkpoints.checkpoint_tracker import CheckpointTracker
from invoke_training.training.shared.data.data_loaders.image_pair_preference_sd_dataloader import (
    build_image_pair_preference_sd_dataloader,
)
from invoke_training.training.shared.optimizer.optimizer_utils import initialize_optimizer
from invoke_training.training.shared.stable_diffusion.lora_checkpoint_utils import save_lora_checkpoint
from invoke_training.training.shared.stable_diffusion.tokenize_captions import tokenize_captions


def train_forward_dpo_without_reference_model(  # noqa: C901
    config: DirectPreferenceOptimizationLoRASDConfig,
    data_batch: dict,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    unet: UNet2DConditionModel,
    ref_text_encoder: CLIPTextModel,
    ref_unet: UNet2DConditionModel,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    """Run the forward training pass for a single data_batch.

    This forward pass is based on 'Diffusion Model Alignment Using Direct Preference Optimization'
    (https://arxiv.org/pdf/2311.12908.pdf). See the "Pseudocode for Training Objective" Appendix section for a helpful
    reference.

    Returns:
        torch.Tensor: Loss
    """
    batch_size = data_batch["image_0"].shape[0]

    # Concatenate image_0 and image_1 images into a single image batch.
    images = torch.concat((data_batch["image_0"], data_batch["image_1"]))

    # Re-order images so that the 'images' batch contains all winner images followed by all loser images.
    w_indices = []
    l_indices = []
    prefer_0 = data_batch["prefer_0"]
    prefer_1 = data_batch["prefer_1"]
    for i in range(batch_size):
        if prefer_0[i] and not prefer_1[i]:
            w_indices.append(i)
            l_indices.append(i + batch_size)
        elif not prefer_0[i] and prefer_1[i]:
            w_indices.append(i + batch_size)
            l_indices.append(i)
        else:
            raise ValueError(f"Encountered image pair with prefer_0={prefer_0[i]} and prefer_1={prefer_1[i]}.")
    images = images[w_indices + l_indices]

    # Update batch_size in case image pairs were filtered due to no-preference.
    batch_size = images.shape[0] // 2

    # Convert images to latent space.
    # The VAE output may have been cached and included in the data_batch. If not, we calculate it here.
    latents = data_batch.get("vae_output", None)
    if latents is None:
        latents = vae.encode(images.to(dtype=weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    # Sample noise that we'll add to the latents.
    # We want to use the same noise for the winning and losing example in each pair, so we generate noise for the
    # winning latents and then repeat it.
    noise = torch.randn_like(latents[:batch_size])
    noise = noise.repeat((2, 1, 1, 1))

    # Sample a random timestep for each image **pair**.
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
    timesteps = timesteps.repeat((2,)).long()

    # Add noise to the latents according to the noise magnitude at each timestep (this is the forward
    # diffusion process).
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Get the text embedding for conditioning (for both the text_encoder and ref_text_encoder).
    # The text_encoder_output may have been cached and included in the data_batch. If not, we calculate it here.
    encoder_hidden_states = data_batch.get("text_encoder_output", None)
    if encoder_hidden_states is None:
        caption_token_ids = tokenize_captions(tokenizer, data_batch["caption"]).to(text_encoder.device)
        encoder_hidden_states = text_encoder(caption_token_ids)[0].to(dtype=weight_dtype)
        ref_encoder_hidden_states = ref_text_encoder(caption_token_ids)[0].to(dtype=weight_dtype)
    encoder_hidden_states = encoder_hidden_states.repeat((2, 1, 1))
    ref_encoder_hidden_states = ref_encoder_hidden_states.repeat((2, 1, 1))

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
    ref_model_pred: torch.Tensor = ref_unet(noisy_latents, timesteps, ref_encoder_hidden_states).sample
    model_pred: torch.Tensor = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    if "loss_weight" in data_batch:
        raise NotImplementedError("loss_weight is not yet supported.")

    target = target.float()
    w_target = target[:batch_size]
    l_target = target[batch_size:]
    model_w_pred = model_pred[:batch_size]
    model_l_pred = model_pred[batch_size:]
    ref_w_pred = ref_model_pred[:batch_size]
    ref_l_pred = ref_model_pred[batch_size:]

    # The pseudo-code from the paper uses `.norm().pow(2)` to calculate the errors. We take the mean over all pixels
    # rather than the sum over all pixels instead. This helps keep the learning rate stable across different image
    # resolutions. It also means that the the recommended settings for beta from the paper are not correct.
    # > model_w_err = (model_w_pred - target).norm().pow(2)
    # > model_l_err = (model_l_pred - target).norm().pow(2)
    # > ref_w_err = (ref_w_pred - target).norm().pow(2)
    # > ref_l_err = (ref_l_pred - target).norm().pow(2)
    model_w_err = torch.nn.functional.mse_loss(model_w_pred, w_target)
    model_l_err = torch.nn.functional.mse_loss(model_l_pred, l_target)
    ref_w_err = torch.nn.functional.mse_loss(ref_w_pred, w_target)
    ref_l_err = torch.nn.functional.mse_loss(ref_l_pred, l_target)

    w_diff = model_w_err - ref_w_err
    l_diff = model_l_err - ref_l_err
    inside_term = -1 * config.beta * (w_diff - l_diff)
    loss = -1 * torch.nn.functional.logsigmoid(inside_term)
    return loss


def run_training(config: DirectPreferenceOptimizationLoRASDConfig):  # noqa: C901
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
    ref_text_encoder = copy.deepcopy(text_encoder)
    ref_unet = copy.deepcopy(unet)

    if config.xformers:
        import xformers  # noqa: F401

        unet.enable_xformers_memory_efficient_attention()
        ref_unet.enable_xformers_memory_efficient_attention()
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
        ref_text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Prepare VAE output cache.
    vae_output_cache_dir_name = None
    if config.cache_vae_outputs:
        raise NotImplementedError("VAE caching is not implemented for Diffusion-DPO training yet.")
        # if config.data_loader.image_transforms.random_flip:
        #     raise ValueError("'cache_vae_outputs' cannot be True if 'random_flip' is True.")
        # if not config.data_loader.image_transforms.center_crop:
        #     raise ValueError("'cache_vae_outputs' cannot be True if 'center_crop' is False.")

        # # We use a temporary directory for the cache. The directory will automatically be cleaned up when
        # # tmp_vae_output_cache_dir is destroyed.
        # tmp_vae_output_cache_dir = tempfile.TemporaryDirectory()
        # vae_output_cache_dir_name = tmp_vae_output_cache_dir.name
        # if accelerator.is_local_main_process:
        #     # Only the main process should populate the cache.
        #     logger.info(f"Generating VAE output cache ('{vae_output_cache_dir_name}').")
        #     vae.to(accelerator.device, dtype=weight_dtype)

        #     data_loader = build_data_loader(
        #         data_loader_config=config.data_loader,
        #         batch_size=config.train_batch_size,
        #         shuffle=False,
        #         sequential_batching=True,
        #     )
        #     cache_vae_outputs(vae_output_cache_dir_name, data_loader, vae)
        # # Move the VAE back to the CPU, because it is not needed for training.
        # vae.to("cpu")
        # accelerator.wait_for_everyone()
    else:
        vae.to(accelerator.device, dtype=weight_dtype)

    unet.to(accelerator.device, dtype=weight_dtype)
    ref_unet.to(accelerator.device, dtype=weight_dtype)

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

    data_loader = build_image_pair_preference_sd_dataloader(
        config=config.data_loader,
        batch_size=config.train_batch_size,
        text_encoder_output_cache_dir=text_encoder_output_cache_dir_name,
        text_encoder_cache_field_to_output_field={"text_encoder_output": "text_encoder_output"},
        vae_output_cache_dir=vae_output_cache_dir_name,
        shuffle=True,
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
                loss = train_forward_dpo_without_reference_model(
                    config=config,
                    data_batch=data_batch,
                    vae=vae,
                    noise_scheduler=noise_scheduler,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    unet=unet,
                    ref_text_encoder=ref_text_encoder,
                    ref_unet=ref_unet,
                    weight_dtype=weight_dtype,
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

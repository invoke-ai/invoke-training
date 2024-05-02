import itertools
import json
import math
import os
import tempfile
import time

import peft
import torch
import torch.utils.data
from accelerate.utils import set_seed
from diffusers import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel

from invoke_training._shared.accelerator.accelerator_utils import (
    get_dtype_from_str,
    initialize_accelerator,
    initialize_logging,
)
from invoke_training._shared.data.samplers.aspect_ratio_bucket_batch_sampler import log_aspect_ratio_buckets
from invoke_training._shared.optimizer.optimizer_utils import initialize_optimizer
from invoke_training._shared.stable_diffusion.model_loading_utils import load_models_sdxl
from invoke_training._shared.stable_diffusion.validation import generate_validation_images_sdxl
from invoke_training._shared.utils.import_xformers import import_xformers
from invoke_training.pipelines.callbacks import PipelineCallbacks
from invoke_training.pipelines.stable_diffusion.lora.train import cache_vae_outputs
from invoke_training.pipelines.stable_diffusion_xl.finetune.config import SdxlFinetuneConfig
from invoke_training.pipelines.stable_diffusion_xl.lora.train import (
    _build_data_loader,
    cache_text_encoder_outputs,
    train_forward,
)


def train(config: SdxlFinetuneConfig, callbacks: list[PipelineCallbacks] | None = None):  # noqa: C901
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

    weight_dtype = get_dtype_from_str(config.weight_dtype)

    logger.info("Loading models.")
    tokenizer_1, tokenizer_2, noise_scheduler, text_encoder_1, text_encoder_2, vae, unet = load_models_sdxl(
        model_name_or_path=config.model,
        hf_variant=config.hf_variant,
        vae_model=config.vae_model,
        base_embeddings=None,
        dtype=weight_dtype,
    )

    if config.xformers:
        import_xformers()

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
            # TODO(ryan): Move cache_text_encoder_outputs to a shared location so that it is not imported from another
            # pipeline.
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

    # Make UNet trainable.
    unet.requires_grad_(True)
    unet.train()
    all_trainable_models = [unet]

    # If mixed_precision is enabled, cast all trainable params to float32.
    if config.mixed_precision != "no":
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

    optimizer = initialize_optimizer(config.optimizer, unet.parameters())

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

    # checkpoint_tracker = CheckpointTracker(
    #     base_dir=ckpt_dir,
    #     prefix="checkpoint",
    #     max_checkpoints=config.max_checkpoints,
    #     # TODO(ryand): Revisit this extension when we add checkpointing.
    #     extension=".safetensors",
    # )

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
            logger.info("TODO: Save checkpoint.")
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
                callbacks=callbacks,
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

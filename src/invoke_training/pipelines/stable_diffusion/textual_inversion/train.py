import json
import logging
import math
import os
import tempfile
import time

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, PreTrainedTokenizer

from invoke_training._shared.accelerator.accelerator_utils import (
    get_dtype_from_str,
    initialize_accelerator,
    initialize_logging,
)
from invoke_training._shared.checkpoints.checkpoint_tracker import CheckpointTracker
from invoke_training._shared.checkpoints.serialization import save_state_dict
from invoke_training._shared.data.data_loaders.textual_inversion_sd_dataloader import (
    build_textual_inversion_sd_dataloader,
)
from invoke_training._shared.data.samplers.aspect_ratio_bucket_batch_sampler import log_aspect_ratio_buckets
from invoke_training._shared.optimizer.optimizer_utils import initialize_optimizer
from invoke_training._shared.stable_diffusion.model_loading_utils import load_models_sd
from invoke_training._shared.stable_diffusion.textual_inversion import (
    initialize_placeholder_tokens_from_initial_embedding,
    initialize_placeholder_tokens_from_initial_phrase,
    initialize_placeholder_tokens_from_initializer_token,
    restore_original_embeddings,
)
from invoke_training._shared.stable_diffusion.validation import generate_validation_images_sd
from invoke_training._shared.utils.import_xformers import import_xformers
from invoke_training.pipelines.callbacks import ModelCheckpoint, ModelType, PipelineCallbacks, TrainingCheckpoint
from invoke_training.pipelines.stable_diffusion.lora.train import cache_vae_outputs, train_forward
from invoke_training.pipelines.stable_diffusion.textual_inversion.config import SdTextualInversionConfig


def _save_ti_embeddings(
    epoch: int,
    step: int,
    text_encoder: CLIPTextModel,
    placeholder_token_ids: list[int],
    accelerator: Accelerator,
    logger: logging.Logger,
    checkpoint_tracker: CheckpointTracker,
    callbacks: list[PipelineCallbacks] | None,
):
    """Save a Textual Inversion checkpoint. Old checkpoints are deleted if necessary to respect the checkpoint_tracker
    limits.
    """
    # Prune checkpoints and get new checkpoint path.
    num_pruned = checkpoint_tracker.prune(1)
    if num_pruned > 0:
        logger.info(f"Pruned {num_pruned} checkpoint(s).")
    save_path = checkpoint_tracker.get_path(epoch=epoch, step=step)

    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {"emb_params": learned_embeds.detach().cpu().to(torch.float32)}

    save_state_dict(learned_embeds_dict, save_path)

    if callbacks is not None:
        for cb in callbacks:
            cb.on_save_checkpoint(
                TrainingCheckpoint(
                    models=[ModelCheckpoint(file_path=save_path, model_type=ModelType.SD1_TEXTUAL_INVERSION)],
                    epoch=epoch,
                    step=step,
                )
            )


def _initialize_placeholder_tokens(
    config: SdTextualInversionConfig,
    tokenizer: CLIPTokenizer,
    text_encoder: PreTrainedTokenizer,
    logger: logging.Logger,
) -> tuple[list[str], list[int]]:
    """Prepare the tokenizer and text_encoder for TI training.

    - Add the placeholder tokens to the tokenizer.
    - Add new token embeddings to the text_encoder for each of the placeholder tokens.
    - Initialize the new token embeddings from either an existing token, or an initial TI embedding file.
    """
    if (
        sum(
            [
                config.initializer_token is not None,
                config.initial_embedding_file is not None,
                config.initial_phrase is not None,
            ]
        )
        != 1
    ):
        raise ValueError(
            "Exactly one of 'initializer_token', 'initial_embedding_file', or 'initial_phrase' should be set."
        )

    if config.initializer_token is not None:
        placeholder_tokens, placeholder_token_ids = initialize_placeholder_tokens_from_initializer_token(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            initializer_token=config.initializer_token,
            placeholder_token=config.placeholder_token,
            num_vectors=config.num_vectors,
            logger=logger,
        )
    elif config.initial_embedding_file is not None:
        placeholder_tokens, placeholder_token_ids = initialize_placeholder_tokens_from_initial_embedding(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            initial_embedding_file=config.initial_embedding_file,
            placeholder_token=config.placeholder_token,
            num_vectors=config.num_vectors,
        )
    elif config.initial_phrase is not None:
        placeholder_tokens, placeholder_token_ids = initialize_placeholder_tokens_from_initial_phrase(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            initial_phrase=config.initial_phrase,
            placeholder_token=config.placeholder_token,
        )
    else:
        raise ValueError(
            "Exactly one of 'initializer_token', 'initial_embedding_file', or 'initial_phrase' should be set."
        )

    return placeholder_tokens, placeholder_token_ids


def train(config: SdTextualInversionConfig, callbacks: list[PipelineCallbacks] | None = None):  # noqa: C901
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

    logger.info("Starting Textual Inversion Training.")
    logger.info(f"Configuration:\n{json.dumps(config.dict(), indent=2, default=str)}")
    logger.info(f"Output dir: '{out_dir}'")

    # Write the configuration to disk.
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config.dict(), f, indent=2, default=str)

    weight_dtype = get_dtype_from_str(config.weight_dtype)

    logger.info("Loading models.")
    tokenizer, noise_scheduler, text_encoder, vae, unet = load_models_sd(
        logger=logger, model_name_or_path=config.model, hf_variant=config.hf_variant, dtype=weight_dtype
    )

    placeholder_tokens, placeholder_token_ids = _initialize_placeholder_tokens(
        config=config, tokenizer=tokenizer, text_encoder=text_encoder, logger=logger
    )
    logger.info(f"Initialized {len(placeholder_tokens)} placeholder tokens: {placeholder_tokens}.")

    # All parameters of the VAE, UNet, and text encoder are currently frozen. Just unfreeze the token embeddings in the
    # text encoder.
    text_encoder.text_model.embeddings.token_embedding.requires_grad_(True)

    if config.gradient_checkpointing:
        # We want to enable gradient checkpointing in the UNet regardless of whether it is being trained.
        unet.enable_gradient_checkpointing()
        # unet must be in train() mode for gradient checkpointing to take effect.
        # At the time of writing, the unet dropout probabilities default to 0, so putting the unet in train mode does
        # not change its forward behavior.
        unet.train()

        # The text_encoder will be put in .train() mode later, so we don't need to worry about that here.
        # Note: There are some weird interactions gradient checkpointing and requires_grad_() when training a
        # text_encoder LoRA. If this code ever gets copied elsewhere, make sure to take a look at how this is handled in
        # other training pipelines.
        text_encoder.gradient_checkpointing_enable()

    if config.xformers:
        import_xformers()

        unet.enable_xformers_memory_efficient_attention()
        vae.enable_xformers_memory_efficient_attention()

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
            # Only the main process should populate the cache.
            logger.info(f"Generating VAE output cache ('{vae_output_cache_dir_name}').")
            vae.to(accelerator.device, dtype=weight_dtype)
            data_loader = build_textual_inversion_sd_dataloader(
                config=config.data_loader,
                placeholder_token=config.placeholder_token,
                batch_size=config.train_batch_size,
                use_masks=config.use_masks,
                shuffle=False,
            )
            cache_vae_outputs(vae_output_cache_dir_name, data_loader, vae)
        # Move the VAE back to the CPU, because it is not needed for training.
        vae.to("cpu")
        accelerator.wait_for_everyone()
    else:
        vae.to(accelerator.device, dtype=weight_dtype)

    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Initialize the optimizer to only optimize the token embeddings.
    optimizer = initialize_optimizer(config.optimizer, text_encoder.get_input_embeddings().parameters())

    data_loader = build_textual_inversion_sd_dataloader(
        config=config.data_loader,
        placeholder_token=config.placeholder_token,
        batch_size=config.train_batch_size,
        use_masks=config.use_masks,
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

    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, data_loader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, data_loader, lr_scheduler
    )

    prepared_result: tuple[
        CLIPTextModel, torch.optim.Optimizer, torch.utils.data.DataLoader, torch.optim.lr_scheduler.LRScheduler
    ] = accelerator.prepare(text_encoder, optimizer, data_loader, lr_scheduler)
    text_encoder, optimizer, data_loader, lr_scheduler = prepared_result

    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion_training")
        # Tensorboard uses markdown formatting, so we wrap the config json in a code block.
        accelerator.log({"configuration": f"```json\n{json.dumps(config.dict(), indent=2, default=str)}\n```\n"})

    checkpoint_tracker = CheckpointTracker(
        base_dir=ckpt_dir,
        prefix="checkpoint",
        extension=".safetensors",
        max_checkpoints=config.max_checkpoints,
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

    # Keep original embeddings as reference.
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

    def save_checkpoint(num_completed_epochs: int, num_completed_steps: int):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            _save_ti_embeddings(
                epoch=num_completed_epochs,
                step=num_completed_steps,
                text_encoder=text_encoder,
                placeholder_token_ids=placeholder_token_ids,
                accelerator=accelerator,
                logger=logger,
                checkpoint_tracker=checkpoint_tracker,
                callbacks=callbacks,
            )
        accelerator.wait_for_everyone()

    def validate(num_completed_epochs: int, num_completed_steps: int):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            generate_validation_images_sd(
                epoch=num_completed_epochs,
                step=num_completed_steps,
                out_dir=out_dir,
                accelerator=accelerator,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                noise_scheduler=noise_scheduler,
                unet=unet,
                config=config,
                logger=logger,
                callbacks=callbacks,
            )
        accelerator.wait_for_everyone()

    for epoch in range(first_epoch, num_train_epochs):
        text_encoder.train()

        train_loss = 0.0
        for data_batch_idx, data_batch in enumerate(data_loader):
            with accelerator.accumulate(text_encoder):
                loss = train_forward(
                    config=config,
                    data_batch=data_batch,
                    vae=vae,
                    noise_scheduler=noise_scheduler,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    unet=unet,
                    weight_dtype=weight_dtype,
                    use_masks=config.use_masks,
                    min_snr_gamma=config.min_snr_gamma,
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                # TODO(ryand): Test that this works properly with distributed training.
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients and config.max_grad_norm is not None:
                    # TODO(ryand): I copied this from another pipeline. Should probably just clip the trainable params.
                    params_to_clip = text_encoder.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Make sure we don't update any embedding weights besides the newly-added token(s).
                # TODO(ryand): Should we only do this if accelerator.sync_gradients?
                restore_original_embeddings(
                    tokenizer=tokenizer,
                    placeholder_token_ids=placeholder_token_ids,
                    accelerator=accelerator,
                    text_encoder=text_encoder,
                    orig_embeds_params=orig_embeds_params,
                )

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                completed_epochs = epoch if (data_batch_idx + 1) < len(data_loader) else epoch + 1
                log = {"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}

                if config.optimizer.optimizer_type == "Prodigy":
                    # TODO(ryand): Test Prodigy logging.
                    log["lr/d*lr"] = optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]

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

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
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

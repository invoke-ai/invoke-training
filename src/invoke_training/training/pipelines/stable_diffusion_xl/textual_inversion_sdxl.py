import json
import logging
import math
import os
import tempfile
import time

import torch
import torch.utils.data
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPPreTrainedModel, CLIPTextModel, CLIPTokenizer, PreTrainedTokenizer

from invoke_training.config.pipelines.finetune_lora_and_ti_config import FinetuneLoraAndTiSdxlConfig
from invoke_training.config.pipelines.textual_inversion_config import TextualInversionSDXLConfig
from invoke_training.training._shared.accelerator.accelerator_utils import (
    get_mixed_precision_dtype,
    initialize_accelerator,
    initialize_logging,
)
from invoke_training.training._shared.checkpoints.checkpoint_tracker import CheckpointTracker
from invoke_training.training._shared.checkpoints.serialization import save_state_dict
from invoke_training.training._shared.data.data_loaders.textual_inversion_sd_dataloader import (
    build_textual_inversion_sd_dataloader,
)
from invoke_training.training._shared.data.samplers.aspect_ratio_bucket_batch_sampler import log_aspect_ratio_buckets
from invoke_training.training._shared.optimizer.optimizer_utils import initialize_optimizer
from invoke_training.training._shared.stable_diffusion.model_loading_utils import load_models_sdxl
from invoke_training.training._shared.stable_diffusion.textual_inversion import (
    initialize_placeholder_tokens_from_initial_phrase,
    initialize_placeholder_tokens_from_initializer_token,
    restore_original_embeddings,
)
from invoke_training.training._shared.stable_diffusion.validation import generate_validation_images_sdxl
from invoke_training.training.pipelines.stable_diffusion_xl.finetune_lora_sdxl import cache_vae_outputs, train_forward


def _save_ti_embeddings(
    idx: int,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModel,
    placeholder_token_ids_1: list[int],
    placeholder_token_ids_2: list[int],
    accelerator: Accelerator,
    logger: logging.Logger,
    checkpoint_tracker: CheckpointTracker,
):
    """Save a Textual Inversion SDXL checkpoint. Old checkpoints are deleted if necessary to respect the
    checkpoint_tracker limits.
    """
    # Prune checkpoints and get new checkpoint path.
    num_pruned = checkpoint_tracker.prune(1)
    if num_pruned > 0:
        logger.info(f"Pruned {num_pruned} checkpoint(s).")
    save_path = checkpoint_tracker.get_path(idx)

    learned_embeds_1 = (
        accelerator.unwrap_model(text_encoder_1)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids_1) : max(placeholder_token_ids_1) + 1]
    )
    learned_embeds_2 = (
        accelerator.unwrap_model(text_encoder_2)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids_2) : max(placeholder_token_ids_2) + 1]
    )
    learned_embeds_dict = {
        "clip_l": learned_embeds_1.detach().cpu(),
        "clip_g": learned_embeds_2.detach().cpu(),
    }

    save_state_dict(learned_embeds_dict, save_path)


def _initialize_placeholder_tokens(
    config: TextualInversionSDXLConfig | FinetuneLoraAndTiSdxlConfig,
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    text_encoder_1: PreTrainedTokenizer,
    text_encoder_2: PreTrainedTokenizer,
) -> tuple[list[str], list[int], list[int]]:
    """Prepare the tokenizers and text_encoders for TI training.

    - Add the placeholder tokens to the tokenizers.
    - Add new token embeddings to the text_encoders for each of the placeholder tokens.
    - Initialize the new token embeddings from either an existing token, or an initial TI embedding file.
    """

    if (
        sum(
            [
                getattr(config, "initializer_token", None) is not None,
                getattr(config, "initial_embedding_file", None) is not None,
                getattr(config, "initial_phrase", None) is not None,
            ]
        )
        != 1
    ):
        raise ValueError(
            "Exactly one of 'initializer_token', 'initial_embedding_file', or 'initial_phrase' should be set."
        )

    if getattr(config, "initializer_token", None) is not None:
        placeholder_tokens_1, placeholder_token_ids_1 = initialize_placeholder_tokens_from_initializer_token(
            tokenizer=tokenizer_1,
            text_encoder=text_encoder_1,
            initializer_token=config.initializer_token,
            placeholder_token=config.placeholder_token,
            num_vectors=config.num_vectors,
        )
        placeholder_tokens_2, placeholder_token_ids_2 = initialize_placeholder_tokens_from_initializer_token(
            tokenizer=tokenizer_2,
            text_encoder=text_encoder_2,
            initializer_token=config.initializer_token,
            placeholder_token=config.placeholder_token,
            num_vectors=config.num_vectors,
        )
    elif getattr(config, "initial_embedding_file", None) is not None:
        # TODO(ryan)
        raise NotImplementedError("Initializing from an initial embedding is not yet supported for SDXL.")
    elif getattr(config, "initial_phrase", None) is not None:
        placeholder_tokens_1, placeholder_token_ids_1 = initialize_placeholder_tokens_from_initial_phrase(
            tokenizer=tokenizer_1,
            text_encoder=text_encoder_1,
            initial_phrase=config.initial_phrase,
            placeholder_token=config.placeholder_token,
        )
        placeholder_tokens_2, placeholder_token_ids_2 = initialize_placeholder_tokens_from_initial_phrase(
            tokenizer=tokenizer_2,
            text_encoder=text_encoder_2,
            initial_phrase=config.initial_phrase,
            placeholder_token=config.placeholder_token,
        )
    else:
        raise ValueError(
            "Exactly one of 'initializer_token', 'initial_embedding_file', or 'initial_phrase' should be set."
        )

    assert placeholder_tokens_1 == placeholder_tokens_2
    return placeholder_tokens_1, placeholder_token_ids_1, placeholder_token_ids_2


def run_training(config: TextualInversionSDXLConfig):  # noqa: C901
    # Create a timestamped directory for all outputs.
    out_dir = os.path.join(config.output.base_output_dir, f"{time.time()}")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir)

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
    tokenizer_1, tokenizer_2, noise_scheduler, text_encoder_1, text_encoder_2, vae, unet = load_models_sdxl(
        model_name_or_path=config.model, hf_variant=config.hf_variant, vae_model=config.vae_model
    )

    placeholder_tokens, placeholder_token_ids_1, placeholder_token_ids_2 = _initialize_placeholder_tokens(
        config=config,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
    )
    logger.info(f"Initialized {len(placeholder_tokens)} placeholder tokens: {placeholder_tokens}.")

    # All parameters of the VAE, UNet, and text encoder are currently frozen. Just unfreeze the token embeddings in the
    # text encoders.
    text_encoder_1.text_model.embeddings.token_embedding.requires_grad_(True)
    text_encoder_2.text_model.embeddings.token_embedding.requires_grad_(True)

    if config.gradient_checkpointing:
        # We want to enable gradient checkpointing in the UNet regardless of whether it is being trained.
        unet.enable_gradient_checkpointing()
        # unet must be in train() mode for gradient checkpointing to take effect.
        # At the time of writing, the unet dropout probabilities default to 0, so putting the unet in train mode does
        # not change its forward behavior.
        unet.train()
        for te in [text_encoder_1, text_encoder_2]:
            # The text_encoder will be put in .train() mode later, so we don't need to worry about that here.
            # Note: There are some weird interactions gradient checkpointing and requires_grad_() when training a
            # text_encoder LoRA. If this code ever gets copied elsewhere, make sure to take a look at how this is
            # handled in other training pipelines.
            te.gradient_checkpointing_enable()

    if config.xformers:
        import xformers  # noqa: F401

        unet.enable_xformers_memory_efficient_attention()
        vae.enable_xformers_memory_efficient_attention()

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
            # Only the main process should to populate the cache.
            logger.info(f"Generating VAE output cache ('{vae_output_cache_dir_name}').")
            vae.to(accelerator.device, dtype=weight_dtype)
            data_loader = build_textual_inversion_sd_dataloader(
                config=config.data_loader,
                placeholder_token=config.placeholder_token,
                batch_size=config.train_batch_size,
                shuffle=False,
            )
            cache_vae_outputs(vae_output_cache_dir_name, data_loader, vae)
        # Move the VAE back to the CPU, because it is not needed for training.
        vae.to("cpu")
        accelerator.wait_for_everyone()
    else:
        vae.to(accelerator.device, dtype=weight_dtype)

    # For mixed precision training, we cast all non-trainable weights (unet, vae) to half-precision as these weights are
    # only used for inference, keeping weights in full precision is not required.
    unet.to(accelerator.device, dtype=weight_dtype)

    # Initialize the optimizer to only optimize the token embeddings.
    trainable_param_groups = [
        {"params": text_encoder_1.get_input_embeddings().parameters()},
        {"params": text_encoder_2.get_input_embeddings().parameters()},
    ]
    optimizer = initialize_optimizer(config.optimizer, trainable_param_groups)
    trainable_models = torch.nn.ModuleDict({"text_encoder_1": text_encoder_1, "text_encoder_2": text_encoder_2})

    data_loader = build_textual_inversion_sd_dataloader(
        config=config.data_loader,
        placeholder_token=config.placeholder_token,
        batch_size=config.train_batch_size,
        vae_output_cache_dir=vae_output_cache_dir_name,
    )

    log_aspect_ratio_buckets(logger=logger, batch_sampler=data_loader.batch_sampler)

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
        CLIPPreTrainedModel,
        CLIPPreTrainedModel,
        torch.optim.Optimizer,
        torch.utils.data.DataLoader,
        torch.optim.lr_scheduler.LRScheduler,
    ] = accelerator.prepare(text_encoder_1, text_encoder_2, optimizer, data_loader, lr_scheduler)
    text_encoder_1, text_encoder_2, optimizer, data_loader, lr_scheduler = prepared_result

    # Calculate the number of epochs and total training steps. A "step" represents a single weight update operation
    # (i.e. takes into account gradient accumulation steps).
    # math.ceil(...) is used in calculating the num_steps_per_epoch, because by default an optimizer step is taken when
    # the end of the dataloader is reached, even if gradient_accumulation_steps hasn't been reached.
    num_steps_per_epoch = math.ceil(len(data_loader) / config.gradient_accumulation_steps)
    num_train_epochs = math.ceil(config.max_train_steps / num_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion_training")
        # Tensorboard uses markdown formatting, so we wrap the config json in a code block.
        accelerator.log({"configuration": f"```json\n{json.dumps(config.dict(), indent=2, default=str)}\n```\n"})

    epoch_checkpoint_tracker = CheckpointTracker(
        base_dir=ckpt_dir,
        prefix="checkpoint_epoch",
        extension=f".{config.output.save_model_as}",
        max_checkpoints=config.max_checkpoints,
    )

    step_checkpoint_tracker = CheckpointTracker(
        base_dir=ckpt_dir,
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

    # Keep original embeddings as reference.
    with torch.no_grad():
        orig_embeds_params_1 = accelerator.unwrap_model(text_encoder_1).get_input_embeddings().weight.data.clone()
        orig_embeds_params_2 = accelerator.unwrap_model(text_encoder_2).get_input_embeddings().weight.data.clone()

    for epoch in range(first_epoch, num_train_epochs):
        text_encoder_1.train()
        text_encoder_2.train()

        train_loss = 0.0
        for data_batch in data_loader:
            with accelerator.accumulate(trainable_models):
                loss = train_forward(
                    accelerator,
                    data_batch,
                    vae,
                    noise_scheduler,
                    tokenizer_1,
                    tokenizer_2,
                    text_encoder_1,
                    text_encoder_2,
                    unet,
                    weight_dtype,
                    config.data_loader.image_transforms.resolution,
                    config.prediction_type,
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                # TODO(ryand): Test that this works properly with distributed training.
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps

                # Backpropagate.
                accelerator.backward(loss)
                if accelerator.sync_gradients and config.max_grad_norm is not None:
                    # TODO(ryand): I copied this from another pipeline. Should probably just clip the trainable params.
                    params_to_clip = trainable_models.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Make sure we don't update any embedding weights besides the newly-added token(s).
                # TODO(ryand): Should we only do this if accelerator.sync_gradients?
                restore_original_embeddings(
                    tokenizer=tokenizer_1,
                    placeholder_token_ids=placeholder_token_ids_1,
                    accelerator=accelerator,
                    text_encoder=text_encoder_1,
                    orig_embeds_params=orig_embeds_params_1,
                )
                restore_original_embeddings(
                    tokenizer=tokenizer_2,
                    placeholder_token_ids=placeholder_token_ids_2,
                    accelerator=accelerator,
                    text_encoder=text_encoder_2,
                    orig_embeds_params=orig_embeds_params_2,
                )

            # Checks if the accelerator has performed an optimization step behind the scenes.
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                log = {"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}

                if config.optimizer.optimizer.optimizer_type == "Prodigy":
                    # TODO(ryand): Test Prodigy logging.
                    log["lr/d*lr"] = optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]

                accelerator.log(log, step=global_step)
                train_loss = 0.0

                if config.save_every_n_steps is not None and (global_step + 1) % config.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        _save_ti_embeddings(
                            idx=global_step + 1,
                            text_encoder_1=text_encoder_1,
                            text_encoder_2=text_encoder_2,
                            placeholder_token_ids_1=placeholder_token_ids_1,
                            placeholder_token_ids_2=placeholder_token_ids_2,
                            accelerator=accelerator,
                            logger=logger,
                            checkpoint_tracker=step_checkpoint_tracker,
                        )

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
                _save_ti_embeddings(
                    idx=epoch + 1,
                    text_encoder_1=text_encoder_1,
                    text_encoder_2=text_encoder_2,
                    placeholder_token_ids_1=placeholder_token_ids_1,
                    placeholder_token_ids_2=placeholder_token_ids_2,
                    accelerator=accelerator,
                    logger=logger,
                    checkpoint_tracker=epoch_checkpoint_tracker,
                )
                # TODO(ryand): This doesn't seem right, but it's done this way in most of the training pipelines. Should
                # probably sync before and after saving. (Or maybe accelerate offers a context manager to handle this?)
                accelerator.wait_for_everyone()

        # Generate validation images every n epochs.
        if len(config.validation_prompts) > 0 and (epoch + 1) % config.validate_every_n_epochs == 0:
            if accelerator.is_main_process:
                generate_validation_images_sdxl(
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

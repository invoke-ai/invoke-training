import json
import logging
import math
import os
import tempfile
import time

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from invoke_training.training.config.textual_inversion_config import (
    TextualInversionConfig,
)
from invoke_training.training.finetune_lora.finetune_lora_sd import (
    cache_vae_outputs,
    generate_validation_images,
    train_forward,
)
from invoke_training.training.shared.accelerator_utils import (
    get_mixed_precision_dtype,
    initialize_accelerator,
    initialize_logging,
)
from invoke_training.training.shared.checkpoint_tracker import CheckpointTracker
from invoke_training.training.shared.data.data_loaders.textual_inversion_sd_dataloader import (
    build_textual_inversion_sd_dataloader,
)
from invoke_training.training.shared.model_loading_utils import (
    PipelineVersionEnum,
    load_pipeline,
)
from invoke_training.training.shared.optimizer_utils import initialize_optimizer
from invoke_training.training.shared.serialization import (
    load_state_dict,
    save_state_dict,
)


def load_models(
    config: TextualInversionConfig,
) -> tuple[CLIPTokenizer, DDPMScheduler, CLIPTextModel, AutoencoderKL, UNet2DConditionModel]:
    """Load all models required for training from disk, transfer them to the target training device.

    Args:
        config (FinetuneLoRAConfig): The LoRA training run config.
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


def save_ti_embeddings(
    idx: int,
    text_encoder: CLIPTextModel,
    placeholder_token_ids: list[int],
    accelerator: Accelerator,
    placeholder_token: str,
    logger: logging.Logger,
    checkpoint_tracker: CheckpointTracker,
):
    """Save a Textual Inversion checkpoint. Old checkpoints are deleted if necessary to respect the checkpoint_tracker
    limits.

    Args:
        idx (int): The checkpoint index (typically step count or epoch).
        text_encoder (CLIPTextModel): The text encoder being trained.
        placeholder_token_ids (list[int]): The placeholder token ids.
        accelerator (Accelerator): The accelerator context (used to unwrap the text_encoder).
        placeholder_token (str): The placeholder token string.
        logger (logging.Logger): Logger.
        checkpoint_tracker (CheckpointTracker): The checkpoint tracker.
    """
    # Prune checkpoints and get new checkpoint path.
    num_pruned = checkpoint_tracker.prune(1)
    if num_pruned > 0:
        logger.info(f"Pruned {num_pruned} checkpoint(s).")
    save_path = checkpoint_tracker.get_path(idx)

    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}

    save_state_dict(learned_embeds_dict, save_path)


def _initialize_placeholder_tokens_from_initializer_token(
    tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, initializer_token: str, placeholder_tokens: list[str]
):
    # Convert the initializer_token and placeholder_token to token ids.
    initializer_token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    if len(initializer_token_ids) > 1:
        raise ValueError(
            f"The initializer_token '{initializer_token}' gets tokenized to {len(initializer_token_ids)} tokens."
            " Choose a different initializer that maps to a single token."
        )
    initializer_token_id = initializer_token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Initialize the newly-added placeholder token(s) with the embeddings of the initializer token.
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    return placeholder_token_ids


def _initialize_placeholder_tokens_from_initial_embedding(
    tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, initial_embedding_file: str, placeholder_tokens: list[str]
):
    base_placeholder_token = placeholder_tokens[0]

    state_dict = load_state_dict(initial_embedding_file)
    if base_placeholder_token not in state_dict:
        raise ValueError(
            f"The initial embedding at '{initial_embedding_file}' does not contain an embedding for placeholder token "
            f"'{base_placeholder_token}'."
        )

    embeddings = state_dict[base_placeholder_token]
    if embeddings.shape[0] != len(placeholder_tokens):
        raise ValueError(
            f"The number of initial embeddings in '{initial_embedding_file}' ({embeddings.shape[0]}) does not match "
            f"the number of placeholder tokens ({len(placeholder_tokens)})."
        )

    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Initialize the newly-added placeholder token(s) with the loaded embeddings.
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for i, token_id in enumerate(placeholder_token_ids):
            token_embeds[token_id] = embeddings[i].clone()

    return placeholder_token_ids


def run_training(config: TextualInversionConfig):  # noqa: C901
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

    logger.info("Starting Textual Inversion Training.")
    logger.info(f"Configuration:\n{json.dumps(config.dict(), indent=2, default=str)}")
    logger.info(f"Output dir: '{out_dir}'")

    # Write the configuration to disk.
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config.dict(), f, indent=2, default=str)

    weight_dtype = get_mixed_precision_dtype(accelerator)

    logger.info("Loading models.")
    tokenizer, noise_scheduler, text_encoder, vae, unet = load_models(config)

    # Add the placeholder token(s) to the tokenizer.
    placeholder_tokens = [config.placeholder_token]
    if config.num_vectors < 1:
        raise ValueError(f"num_vectors must be >1, but is '{config.num_vectors}'.")
    # Add dummy placeholder tokens if num_vectors > 1.
    for i in range(1, config.num_vectors):
        placeholder_tokens.append(f"{config.placeholder_token}_{i}")

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != config.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token '{config.placeholder_token}'. Please pass a different"
            " 'placeholder_token' that is not already in the tokenizer."
        )
    # Resize the token embeddings as we have added new special tokens to the tokenizer.
    text_encoder.resize_token_embeddings(len(tokenizer))

    if config.initializer_token is not None and config.initial_embedding_file is not None:
        raise ValueError(
            "Both 'initializer_token' and 'initial_embedding_file' are non-None. Only one of these fields should be "
            "set."
        )
    elif config.initializer_token is not None:
        placeholder_token_ids = _initialize_placeholder_tokens_from_initializer_token(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            initializer_token=config.initializer_token,
            placeholder_tokens=placeholder_tokens,
        )
    elif config.initial_embedding_file is not None:
        placeholder_token_ids = _initialize_placeholder_tokens_from_initial_embedding(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            initial_embedding_file=config.initial_embedding_file,
            placeholder_tokens=placeholder_tokens,
        )
    else:
        raise ValueError(
            "Both 'initializer_token' and 'initial_embedding_file' are None. One of these fields must be set."
        )

    # All parameters of the VAE, UNet, and text encoder are currently frozen. Just unfreeze the token embeddings in the
    # text encoder.
    text_encoder.text_model.embeddings.token_embedding.requires_grad_(True)

    if config.gradient_checkpointing:
        # unet must be in train() mode for gradient checkpointing to take effect.
        # At the time of writing, the unet dropout probabilities default to 0, so putting the unet in train mode does
        # not change its forward behavior.
        unet.train()
        unet.enable_gradient_checkpointing()

        # The text_encoder will be put in .train() mode later, so we don't need to worry about that here.
        # Note: There are some weird interactions gradient checkpointing and requires_grad_() when training a
        # text_encoder LoRA. If this code ever gets copied elsewhere, make sure to take a look at how this is handled in
        # other training pipelines.
        text_encoder.gradient_checkpointing_enable()

    if config.xformers:
        import xformers  # noqa: F401

        unet.enable_xformers_memory_efficient_attention()
        vae.enable_xformers_memory_efficient_attention()

    # Prepare VAE output cache.
    vae_output_cache_dir_name = None
    if config.cache_vae_outputs:
        if config.dataset.image_transforms.random_flip:
            raise ValueError("'cache_vae_outputs' cannot be True if 'random_flip' is True.")
        if not config.dataset.image_transforms.center_crop:
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
                config=config.dataset,
                placeholder_str=config.placeholder_token,
                learnable_property=config.learnable_property,
                batch_size=config.train_batch_size,
                shuffle=False,
            )
            cache_vae_outputs(vae_output_cache_dir_name, data_loader, vae)
        # Move the VAE back to the CPU, because it is not needed for training.
        vae.to("cpu")
        accelerator.wait_for_everyone()
    else:
        vae.to(accelerator.device, dtype=weight_dtype)

    # Initialize the optimizer to only optimize the token embeddings.
    optimizer = initialize_optimizer(config.optimizer, text_encoder.get_input_embeddings().parameters())

    data_loader = build_textual_inversion_sd_dataloader(
        config=config.dataset,
        placeholder_str=config.placeholder_token,
        learnable_property=config.learnable_property,
        batch_size=config.train_batch_size,
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

    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, data_loader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, data_loader, lr_scheduler
    )

    prepared_result: tuple[
        CLIPTextModel, torch.optim.Optimizer, torch.utils.data.DataLoader, torch.optim.lr_scheduler.LRScheduler
    ] = accelerator.prepare(text_encoder, optimizer, data_loader, lr_scheduler)
    text_encoder, optimizer, data_loader, lr_scheduler = prepared_result

    # For mixed precision training, we cast all non-trainable weights (unet, vae) to half-precision as these weights are
    # only used for inference, keeping weights in full precision is not required.
    # TODO(ryand): Re-think when to do this when adding VAE output caching.
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

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

    # Keep original embeddings as reference.
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

    for epoch in range(first_epoch, num_train_epochs):
        text_encoder.train()

        train_loss = 0.0
        for data_batch in data_loader:
            with accelerator.accumulate(text_encoder):
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

                accelerator.backward(loss)
                if accelerator.sync_gradients and config.max_grad_norm is not None:
                    # TODO(ryand): I copied this from another pipeline. Should probably just clip the trainable params.
                    params_to_clip = text_encoder.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Let's make sure we don't update any embedding weights besides the newly added token.
                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False
                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                log = {"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()}

                if config.optimizer.optimizer.optimizer_type == "Prodigy":
                    # TODO(ryand): Test Prodigy logging.
                    log["lr/d*lr"] = optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]

                accelerator.log(log, step=global_step)
                train_loss = 0.0

                if config.save_every_n_steps is not None and (global_step + 1) % config.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_ti_embeddings(
                            idx=global_step + 1,
                            text_encoder=text_encoder,
                            placeholder_token_ids=placeholder_token_ids,
                            accelerator=accelerator,
                            placeholder_token=config.placeholder_token,
                            logger=logger,
                            checkpoint_tracker=step_checkpoint_tracker,
                        )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= config.max_train_steps:
                break

        # Save a checkpoint every n epochs.
        if config.save_every_n_epochs is not None and (epoch + 1) % config.save_every_n_epochs == 0:
            if accelerator.is_main_process:
                save_ti_embeddings(
                    idx=epoch + 1,
                    text_encoder=text_encoder,
                    placeholder_token_ids=placeholder_token_ids,
                    accelerator=accelerator,
                    placeholder_token=config.placeholder_token,
                    logger=logger,
                    checkpoint_tracker=epoch_checkpoint_tracker,
                )
                # TODO(ryand): This doesn't seem right, but it's done this way in most of the training pipelines. Should
                # probably sync before and after saving. (Or maybe accelerate offers a context manager to handle this?)
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

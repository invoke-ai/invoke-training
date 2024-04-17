import itertools
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Literal

import peft
import torch
import torch.utils.data
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel

from invoke_training._shared.accelerator.accelerator_utils import (
    get_mixed_precision_dtype,
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
from invoke_training._shared.stable_diffusion.lora_checkpoint_utils import (
    TEXT_ENCODER_TARGET_MODULES,
    UNET_TARGET_MODULES,
    save_sdxl_kohya_checkpoint,
    save_sdxl_peft_checkpoint,
)
from invoke_training._shared.stable_diffusion.model_loading_utils import load_models_sdxl
from invoke_training._shared.stable_diffusion.textual_inversion import restore_original_embeddings
from invoke_training._shared.stable_diffusion.validation import generate_validation_images_sdxl
from invoke_training.pipelines.stable_diffusion_xl.lora.train import train_forward
from invoke_training.pipelines.stable_diffusion_xl.lora_and_textual_inversion.config import (
    SdxlLoraAndTextualInversionConfig,
)
from invoke_training.pipelines.stable_diffusion_xl.textual_inversion.train import _initialize_placeholder_tokens


def _save_sdxl_lora_and_ti_checkpoint(
    config: SdxlLoraAndTextualInversionConfig,
    idx: int,
    unet: peft.PeftModel | None,
    text_encoder_1: peft.PeftModel | None,
    text_encoder_2: peft.PeftModel | None,
    placeholder_token_ids_1: list[int],
    placeholder_token_ids_2: list[int],
    accelerator: Accelerator,
    logger: logging.Logger,
    checkpoint_tracker: CheckpointTracker,
    lora_checkpoint_format: Literal["invoke_peft", "kohya"],
):
    # Prune checkpoints and get new checkpoint path.
    num_pruned = checkpoint_tracker.prune(1)
    if num_pruned > 0:
        logger.info(f"Pruned {num_pruned} checkpoint(s).")
    save_path = checkpoint_tracker.get_path(idx)

    if lora_checkpoint_format == "invoke_peft":
        save_sdxl_peft_checkpoint(
            Path(save_path),
            unet=unet if config.train_unet else None,
            text_encoder_1=text_encoder_1 if config.train_text_encoder else None,
            text_encoder_2=text_encoder_2 if config.train_text_encoder else None,
        )
    elif lora_checkpoint_format == "kohya":
        save_sdxl_kohya_checkpoint(
            Path(save_path) / "lora.safetensors",
            unet=unet if config.train_unet else None,
            text_encoder_1=text_encoder_1 if config.train_text_encoder else None,
            text_encoder_2=text_encoder_2 if config.train_text_encoder else None,
        )
    else:
        raise ValueError(f"Unsupported lora_checkpoint_format: '{lora_checkpoint_format}'.")

    if config.train_ti:
        ti_checkpoint_path = Path(save_path) / "embeddings.safetensors"
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
        save_state_dict(learned_embeds_dict, ti_checkpoint_path)


def train(config: SdxlLoraAndTextualInversionConfig):  # noqa: C901
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
    )

    if config.xformers:
        import xformers  # noqa: F401

        # TODO(ryand): There is a known issue if xformers is enabled when training in mixed precision where xformers
        # will fail because Q, K, V have different dtypes.
        unet.enable_xformers_memory_efficient_attention()
        vae.enable_xformers_memory_efficient_attention()

    # Prepare text encoder output cache.
    # text_encoder_output_cache_dir_name = None
    if config.cache_text_encoder_outputs:
        raise NotImplementedError("Caching text encoder outputs is not yet supported.")
    else:
        text_encoder_1.to(accelerator.device, dtype=weight_dtype)
        text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # Prepare VAE output cache.
    vae_output_cache_dir_name = None
    if config.cache_vae_outputs:
        raise NotImplementedError("Caching VAE outputs is not yet supported.")
    else:
        vae.to(accelerator.device, dtype=weight_dtype)

    unet.to(accelerator.device, dtype=weight_dtype)

    # Add LoRA layers to the models being trained.
    trainable_param_groups = []
    all_trainable_models: set[torch.nn.Module] = set()

    def inject_lora_layers(model, lora_config: peft.LoraConfig, lr: float) -> peft.PeftModel:
        peft_model = peft.get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()

        # Populate `trainable_param_groups`, to be passed to the optimizer.
        param_group = {"params": list(filter(lambda p: p.requires_grad, peft_model.parameters())), "lr": lr}
        trainable_param_groups.append(param_group)

        # Populate all_trainable_models.
        all_trainable_models.add(peft_model)
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

    if config.train_ti:
        # TODO(ryand): Move this private function to a shared location.
        placeholder_tokens, placeholder_token_ids_1, placeholder_token_ids_2 = _initialize_placeholder_tokens(
            config=config,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
        )
        logger.info(f"Initialized {len(placeholder_tokens)} placeholder tokens: {placeholder_tokens}.")

        # Unfreeze the token embeddings in the text encoders.
        text_encoder_1.text_model.embeddings.token_embedding.requires_grad_(True)
        text_encoder_2.text_model.embeddings.token_embedding.requires_grad_(True)

        all_trainable_models.add(text_encoder_1)
        all_trainable_models.add(text_encoder_2)

        for te in [text_encoder_1, text_encoder_2]:
            param_group = {
                "params": te.get_input_embeddings().parameters(),
                "lr": config.textual_inversion_learning_rate,
            }
            trainable_param_groups.append(param_group)

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
                # trainable_param_groups has already been populated - this won't change what gets trained.
                te.text_model.embeddings.requires_grad_(True)

    optimizer = initialize_optimizer(config.optimizer, trainable_param_groups)

    data_loader = build_textual_inversion_sd_dataloader(
        config=config.data_loader,
        placeholder_token=config.placeholder_token,
        batch_size=config.train_batch_size,
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
        accelerator.init_trackers("lora_and_ti_training")
        # Tensorboard uses markdown formatting, so we wrap the config json in a code block.
        accelerator.log({"configuration": f"```json\n{json.dumps(config.dict(), indent=2, default=str)}\n```\n"})

    epoch_checkpoint_tracker = CheckpointTracker(
        base_dir=ckpt_dir,
        prefix="checkpoint_epoch",
        max_checkpoints=config.max_checkpoints,
    )

    step_checkpoint_tracker = CheckpointTracker(
        base_dir=ckpt_dir,
        prefix="checkpoint_step",
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

    progress_bar = tqdm(
        range(global_step, num_train_steps),
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    ti_train_steps = num_train_steps
    if config.ti_train_steps_ratio is not None:
        ti_train_steps = math.ceil(num_train_steps * config.ti_train_steps_ratio)
        logger.info(f"The TI training pivot point is set at {ti_train_steps} steps.")

    # Keep original embeddings as reference.
    with torch.no_grad():
        orig_embeds_params_1 = accelerator.unwrap_model(text_encoder_1).get_input_embeddings().weight.data.clone()
        orig_embeds_params_2 = accelerator.unwrap_model(text_encoder_2).get_input_embeddings().weight.data.clone()

    for epoch in range(first_epoch, num_train_epochs):
        # TODO(ryand): Is this necessary?
        text_encoder_1.train()
        text_encoder_2.train()

        train_loss = 0.0
        for data_batch in data_loader:
            if global_step == ti_train_steps and config.train_ti:
                logger.info("Reached TI training pivot point. Setting TI learning rate to 0.0.")
                # TODO(ryand): The TI embeddings continue to be updated slightly by the normalization step in
                # restore_original_embeddings(...). The updates should be very small and converge quickly, so this
                # should be fine. But, at some point we should tidy this up.
                for ti_param_group in optimizer.param_groups[-2:]:
                    # The TI param groups should be the last two param groups. But, this is pretty brittle, so this
                    # assertion adds a bit of safety.
                    assert len(ti_param_group["params"]) == 1
                    ti_param_group["lr"] = 0.0

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
                log = {"train_loss": train_loss}

                lrs = lr_scheduler.get_last_lr()

                # Prepare LR names in the same order that their respective param groups were added to the optimizer.
                # TODO: Do this at the time that we prepare the param groups?
                lr_names = []
                if config.train_unet:
                    lr_names.append("unet")
                if config.train_text_encoder:
                    lr_names.append("text_encoder_1")
                    lr_names.append("text_encoder_2")
                if config.train_ti:
                    lr_names.append("ti_embeddings_1")
                    lr_names.append("ti_embeddings_2")

                for lr_idx, lr_name in enumerate(lr_names):
                    log[f"lr/{lr_name}"] = float(lrs[lr_idx])
                    if config.optimizer.optimizer_type == "Prodigy":
                        log[f"lr/d*lr/{lr_name}"] = (
                            optimizer.param_groups[lr_idx]["d"] * optimizer.param_groups[lr_idx]["lr"]
                        )

                accelerator.log(log, step=global_step)
                train_loss = 0.0

                # global_step represents the *number of completed steps* at this point.
                if config.save_every_n_steps is not None and global_step % config.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        _save_sdxl_lora_and_ti_checkpoint(
                            config=config,
                            idx=global_step,
                            unet=unet,
                            text_encoder_1=text_encoder_1,
                            text_encoder_2=text_encoder_2,
                            placeholder_token_ids_1=placeholder_token_ids_1,
                            placeholder_token_ids_2=placeholder_token_ids_2,
                            accelerator=accelerator,
                            logger=logger,
                            checkpoint_tracker=step_checkpoint_tracker,
                            lora_checkpoint_format=config.lora_checkpoint_format,
                        )

                if (
                    config.validate_every_n_steps is not None
                    and global_step % config.validate_every_n_steps == 0
                    and len(config.validation_prompts) > 0
                ):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        generate_validation_images_sdxl(
                            step=global_step,
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
                            prefix="step",
                        )
            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= num_train_steps:
                break

        # Save a checkpoint every n epochs.
        # (epoch + 1) represents the *number of completed epochs* at this point.
        if config.save_every_n_epochs is not None and (epoch + 1) % config.save_every_n_epochs == 0:
            if accelerator.is_main_process:
                _save_sdxl_lora_and_ti_checkpoint(
                    config=config,
                    idx=epoch + 1,
                    unet=unet,
                    text_encoder_1=text_encoder_1,
                    text_encoder_2=text_encoder_2,
                    placeholder_token_ids_1=placeholder_token_ids_1,
                    placeholder_token_ids_2=placeholder_token_ids_2,
                    accelerator=accelerator,
                    logger=logger,
                    checkpoint_tracker=epoch_checkpoint_tracker,
                    lora_checkpoint_format=config.lora_checkpoint_format,
                )
                accelerator.wait_for_everyone()

        # Generate validation images every n epochs.
        if (
            config.validate_every_n_epochs is not None
            and (epoch + 1) % config.validate_every_n_epochs == 0
            and len(config.validation_prompts) > 0
        ):
            if accelerator.is_main_process:
                generate_validation_images_sdxl(
                    step=epoch + 1,
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
                    prefix="epoch",
                )

    accelerator.end_training()

import logging
import os

import peft
import torch

from invoke_training.core.lora.injection.stable_diffusion import (
    convert_lora_state_dict_to_kohya_format,
)
from invoke_training.training.shared.checkpoints.checkpoint_tracker import CheckpointTracker
from invoke_training.training.shared.checkpoints.serialization import save_state_dict


def save_lora_checkpoint(
    idx: int,
    lora_layers: torch.nn.ModuleDict,
    logger: logging.Logger,
    checkpoint_tracker: CheckpointTracker,
):
    """Save a LoRA checkpoint. Old checkpoints are deleted if necessary to respect the config.max_checkpoints config.

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
        model_kohya_state_dict = convert_lora_state_dict_to_kohya_format(model_state_dict)
        state_dict.update(model_kohya_state_dict)

    save_state_dict(state_dict, save_path)
    # accelerator.save_state(save_path)
    logger.info(f"Saved state to '{save_path}'.")


UNET_PEFT_FILE_NAME = "unet_lora"
TEXT_ENCODER_PEFT_FILE_NAME = "text_encoder_lora"
TEXT_ENCODER_2_PEFT_FILE_NAME = "text_encoder_2_lora"


def save_peft_lora_checkpoint(
    idx: int,
    unet: peft.PeftModel | None,
    text_encoder: peft.PeftModel | None,
    text_encoder_2: peft.PeftModel | None,
    logger: logging.Logger,
    checkpoint_tracker: CheckpointTracker,
):
    # Prune checkpoints and get new checkpoint path.
    num_pruned = checkpoint_tracker.prune(1)
    if num_pruned > 0:
        logger.info(f"Pruned {num_pruned} checkpoint(s).")
    save_path = checkpoint_tracker.get_path(idx)

    files_and_models: list[tuple[str, peft.PeftModel | None]] = [
        (UNET_PEFT_FILE_NAME, unet),
        (TEXT_ENCODER_PEFT_FILE_NAME, text_encoder),
        (TEXT_ENCODER_2_PEFT_FILE_NAME, text_encoder_2),
    ]

    for file_name, peft_model in files_and_models:
        if peft_model is not None:
            peft_model.save_pretrained(os.path.join(save_path, file_name))

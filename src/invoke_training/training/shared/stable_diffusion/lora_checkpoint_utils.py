import logging
from pathlib import Path

import torch

from invoke_training.core.lora.injection.stable_diffusion import (
    convert_invoke_to_kohya_lora_state_dict,
    convert_kohya_to_invoke_lora_state_dict,
)
from invoke_training.training.shared.checkpoints.checkpoint_tracker import CheckpointTracker
from invoke_training.training.shared.checkpoints.serialization import load_state_dict, save_state_dict


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
        model_kohya_state_dict = convert_invoke_to_kohya_lora_state_dict(model_state_dict)
        state_dict.update(model_kohya_state_dict)

    save_state_dict(state_dict, save_path)
    # accelerator.save_state(save_path)
    logger.info(f"Saved state to '{save_path}'.")


def load_lora_checkpoint(lora_layers: torch.nn.ModuleDict, lora_path: str | Path):
    """Load weights from the LoRA checkpoint at lora_path into lora_layers."""
    kohya_state_dict = load_state_dict(lora_path)

    for model_lora_layers in lora_layers.values():
        model_state_dict = model_lora_layers.get_lora_state_dict()
        kohya_state_dict_in_invoke_format = convert_kohya_to_invoke_lora_state_dict(
            invoke_state_dict=model_state_dict, kohya_state_dict=kohya_state_dict
        )

        for key, val in kohya_state_dict_in_invoke_format.items():
            # HACK(ryand): Note that we are copying the values over rather than just the Tensor reference. This is
            # necessary because there are already other references to the tensors.
            model_state_dict[key][...] = val[...]

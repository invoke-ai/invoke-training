import logging
from pathlib import Path

import peft
import torch
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel

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


SD_PEFT_UNET_KEY = "unet"
SD_PEFT_TEXT_ENCODER_KEY = "text_encoder"

SDXL_PEFT_UNET_KEY = "unet"
SDXL_PEFT_TEXT_ENCODER_1_KEY = "text_encoder_1"
SDXL_PEFT_TEXT_ENCODER_2_KEY = "text_encoder_2"


def save_multi_model_peft_checkpoint(checkpoint_dir: Path, models: dict[str, peft.PeftModel]):
    """Save a dict of PeftModels to a checkpoint directory.

    The `models` dict keys are used as the subdirectories for each individual model.

    `load_multi_model_peft_checkpoint(...)` can be used to load the resultant checkpoint.
    """
    for model_key, peft_model in models.items():
        assert isinstance(peft_model, peft.PeftModel)
        peft_model.save_pretrained(str(checkpoint_dir / model_key))


def load_multi_model_peft_checkpoint(
    checkpoint_dir: Path,
    models: dict[str, torch.nn.Module],
    is_trainable: bool = False,
    raise_if_subdir_missing: bool = True,
) -> dict[str, torch.nn.Module]:
    """Load a multi-model PEFT checkpoint that was saved with `save_multi_model_peft_checkpoint(...)`."""
    assert checkpoint_dir.exists()

    out_models = {}
    for model_key, model in models:
        dir_path: Path = checkpoint_dir / model_key
        if dir_path.exists():
            out_models[model_key] = peft.PeftModel.from_pretrained(model, dir_path, is_trainable=is_trainable)
        else:
            if raise_if_subdir_missing:
                raise ValueError(f"'{dir_path}' does not exist.")
            else:
                # Pass through the model unchanged.
                out_models[model_key] = model

    return out_models


def save_sd_peft_checkpoint(checkpoint_dir: Path, unet: peft.PeftModel | None, text_encoder: peft.PeftModel | None):
    models = {}
    if unet is not None:
        models[SD_PEFT_UNET_KEY] = unet
    if text_encoder is not None:
        models[SD_PEFT_TEXT_ENCODER_KEY] = text_encoder

    save_multi_model_peft_checkpoint(checkpoint_dir=checkpoint_dir, models=models)


def load_sd_peft_checkpoint(
    checkpoint_dir: Path, unet: UNet2DConditionModel, text_encoder: CLIPTextModel, is_trainable: bool = False
):
    models = load_multi_model_peft_checkpoint(
        checkpoint_dir=checkpoint_dir,
        models={SD_PEFT_UNET_KEY: unet, SD_PEFT_TEXT_ENCODER_KEY: text_encoder},
        is_trainable=is_trainable,
        raise_if_subdir_missing=False,
    )

    return models[SD_PEFT_UNET_KEY], models[SD_PEFT_TEXT_ENCODER_KEY]


def save_sdxl_peft_checkpoint(
    checkpoint_dir: Path,
    unet: peft.PeftModel | None,
    text_encoder_1: peft.PeftModel | None,
    text_encoder_2: peft.PeftModel | None,
):
    models = {}
    if unet is not None:
        models[SDXL_PEFT_UNET_KEY] = unet
    if text_encoder_1 is not None:
        models[SDXL_PEFT_TEXT_ENCODER_1_KEY] = text_encoder_1
    if text_encoder_2 is not None:
        models[SDXL_PEFT_TEXT_ENCODER_2_KEY] = text_encoder_2

    save_multi_model_peft_checkpoint(checkpoint_dir=checkpoint_dir, models=models)


def load_sdxl_peft_checkpoint(
    checkpoint_dir: Path,
    unet: UNet2DConditionModel,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModel,
    is_trainable: bool = False,
):
    models = load_multi_model_peft_checkpoint(
        checkpoint_dir=checkpoint_dir,
        models={
            SDXL_PEFT_UNET_KEY: unet,
            SDXL_PEFT_TEXT_ENCODER_1_KEY: text_encoder_1,
            SDXL_PEFT_TEXT_ENCODER_2_KEY: text_encoder_2,
        },
        is_trainable=is_trainable,
        raise_if_subdir_missing=False,
    )

    return models[SDXL_PEFT_UNET_KEY], models[SDXL_PEFT_TEXT_ENCODER_1_KEY], models[SDXL_PEFT_TEXT_ENCODER_2_KEY]


def save_sd_lora_checkpoint(
    idx: int,
    unet: peft.PeftModel | None,
    text_encoder: peft.PeftModel | None,
    logger: logging.Logger,
    checkpoint_tracker: CheckpointTracker,
):
    # Prune checkpoints and get new checkpoint path.
    num_pruned = checkpoint_tracker.prune(1)
    if num_pruned > 0:
        logger.info(f"Pruned {num_pruned} checkpoint(s).")
    save_path = checkpoint_tracker.get_path(idx)

    save_sd_peft_checkpoint(Path(save_path), unet=unet, text_encoder=text_encoder)


def save_sdxl_lora_checkpoint(
    idx: int,
    unet: peft.PeftModel | None,
    text_encoder_1: peft.PeftModel | None,
    text_encoder_2: peft.PeftModel | None,
    logger: logging.Logger,
    checkpoint_tracker: CheckpointTracker,
):
    # Prune checkpoints and get new checkpoint path.
    num_pruned = checkpoint_tracker.prune(1)
    if num_pruned > 0:
        logger.info(f"Pruned {num_pruned} checkpoint(s).")
    save_path = checkpoint_tracker.get_path(idx)

    save_sdxl_peft_checkpoint(Path(save_path), unet=unet, text_encoder_1=text_encoder_1, text_encoder_2=text_encoder_2)


# This implementation is based on
# https://github.com/huggingface/peft/blob/8665e2b5719faa4e4b91749ddec09442927b53e0/examples/lora_dreambooth/convert_peft_sd_lora_to_kohya_ss.py#L20
def _convert_peft_state_dict_to_kohya_state_dict(
    lora_config: peft.LoraConfig,
    peft_state_dict: dict[str, torch.Tensor],
    prefix: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    kohya_ss_state_dict = {}
    for peft_key, weight in peft_state_dict.items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(lora_config.lora_alpha).to(dtype)

    return kohya_ss_state_dict


def convert_sd_peft_checkpoint_to_kohya_state_dict(
    in_checkpoint_dir: Path,
    out_checkpoint_file: Path,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Convert SD v1 PEFT models to a Kohya-format LoRA state dict."""
    kohya_state_dict = {}
    for kohya_prefix, peft_model_key in [("lora_unet", SD_PEFT_UNET_KEY), ("lora_te", SD_PEFT_TEXT_ENCODER_KEY)]:
        peft_model_dir = in_checkpoint_dir / peft_model_key

        if peft_model_dir.exists():
            # Note: This logic to load the LoraConfig and weights directly is based on how it is done here:
            # https://github.com/huggingface/peft/blob/8665e2b5719faa4e4b91749ddec09442927b53e0/src/peft/peft_model.py#L672-L689
            # This may need to be updated in the future to support other adapter types (LoKr, LoHa, etc.).
            # Also, I could see this interface breaking in the future.
            lora_config = peft.LoraConfig.from_pretrained(peft_model_dir)
            lora_weights = peft.utils.load_peft_weights(peft_model_dir, device="cpu")

            kohya_state_dict.update(
                _convert_peft_state_dict_to_kohya_state_dict(
                    lora_config=lora_config, peft_state_dict=lora_weights, prefix=kohya_prefix, dtype=dtype
                )
            )

    save_state_dict(kohya_state_dict, out_checkpoint_file)

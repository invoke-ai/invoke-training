import os
from pathlib import Path

import peft
import torch
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel

from invoke_training._shared.checkpoints.serialization import save_state_dict

# Copied from https://github.com/huggingface/peft/blob/8665e2b5719faa4e4b91749ddec09442927b53e0/examples/stable_diffusion/train_dreambooth.py#L49C1-L65C87
FLUX_DIFFUSER_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "proj",
    "proj_in",
    "proj_out",
    "conv",
    "conv1",
    "conv2",
    "conv_shortcut",
    "to_out.0",
    "time_emb_proj",
    "ff.net.2",
]
TEXT_ENCODER_TARGET_MODULES = ["fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj"]

# Module lists copied from diffusers training script.
# These module lists will produce lighter, less expressive, LoRA models than the non-light versions.
UNET_TARGET_MODULES_LIGHT = ["to_k", "to_q", "to_v", "to_out.0"]
TEXT_ENCODER_TARGET_MODULES_LIGHT = ["q_proj", "k_proj", "v_proj", "out_proj"]

FLUX_DIFFUSER_PEFT_UNET_KEY = "transformer"
FLUX_DIFFUSER_PEFT_TEXT_ENCODER_1_KEY = "text_encoder_1"
FLUX_DIFFUSER_PEFT_TEXT_ENCODER_2_KEY = "text_encoder_2"


def save_multi_model_peft_checkpoint(checkpoint_dir: Path | str, models: dict[str, peft.PeftModel]):
    """Save a dict of PeftModels to a checkpoint directory.

    The `models` dict keys are used as the subdirectories for each individual model.

    `load_multi_model_peft_checkpoint(...)` can be used to load the resultant checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)
    for model_key, peft_model in models.items():
        assert isinstance(peft_model, peft.PeftModel)

        # HACK(ryand): PeftModel.save_pretrained(...) expects the config to have a "_name_or_path" entry. For now, we
        # set this to None here. This should be fixed upstream in PEFT.
        if (
            hasattr(peft_model, "config")
            and isinstance(peft_model.config, dict)
            and "_name_or_path" not in peft_model.config
        ):
            peft_model.config["_name_or_path"] = None

        peft_model.save_pretrained(str(checkpoint_dir / model_key))


def load_multi_model_peft_checkpoint(
    checkpoint_dir: Path | str,
    models: dict[str, torch.nn.Module],
    is_trainable: bool = False,
    raise_if_subdir_missing: bool = True,
) -> dict[str, torch.nn.Module]:
    """Load a multi-model PEFT checkpoint that was saved with `save_multi_model_peft_checkpoint(...)`."""
    checkpoint_dir = Path(checkpoint_dir)
    assert checkpoint_dir.exists()

    out_models = {}
    for model_key, model in models.items():
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


def load_flux_peft_checkpoint(
    checkpoint_dir: Path | str, diffuser: UNet2DConditionModel, text_encoder_1: CLIPTextModel, text_encoder_2: CLIPTextModel, is_trainable: bool = False
):
    models = load_multi_model_peft_checkpoint(
        checkpoint_dir=checkpoint_dir,
        models={
            FLUX_DIFFUSER_PEFT_UNET_KEY: diffuser,
            FLUX_DIFFUSER_PEFT_TEXT_ENCODER_1_KEY: text_encoder_1,
            FLUX_DIFFUSER_PEFT_TEXT_ENCODER_2_KEY: text_encoder_2,
        },
        is_trainable=is_trainable,
        raise_if_subdir_missing=False,
    )

    return models[SD_PEFT_UNET_KEY], models[SD_PEFT_TEXT_ENCODER_KEY]


def save_flux_peft_checkpoint(
    checkpoint_dir: Path | str,
    diffuser: peft.PeftModel | None,
    text_encoder_1: peft.PeftModel | None,
    text_encoder_2: peft.PeftModel | None,
):
    models = {}
    if unet is not None:
        models[FLUX_PEFT_UNET_KEY] = diffuser
    if text_encoder_1 is not None:
        models[FLUX_PEFT_TEXT_ENCODER_1_KEY] = text_encoder_1
    if text_encoder_2 is not None:
        models[FLUX_PEFT_TEXT_ENCODER_2_KEY] = text_encoder_2

    save_multi_model_peft_checkpoint(checkpoint_dir=checkpoint_dir, models=models)


def load_sdxl_peft_checkpoint(
    checkpoint_dir: Path | str,
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
            alpha_key = f"{kohya_key.split('.')[0]}.alpha"
            kohya_ss_state_dict[alpha_key] = torch.tensor(lora_config.lora_alpha).to(dtype)

    return kohya_ss_state_dict


def _convert_peft_models_to_kohya_state_dict(
    kohya_prefixes: list[str], models: list[peft.PeftModel]
) -> dict[str, torch.Tensor]:
    kohya_state_dict = {}
    default_adapter_name = "default"

    for kohya_prefix, peft_model in zip(kohya_prefixes, models, strict=True):
        lora_config = peft_model.peft_config[default_adapter_name]
        assert isinstance(lora_config, peft.LoraConfig)

        peft_state_dict = peft.get_peft_model_state_dict(peft_model, adapter_name=default_adapter_name)

        kohya_state_dict.update(
            _convert_peft_state_dict_to_kohya_state_dict(
                lora_config=lora_config,
                peft_state_dict=peft_state_dict,
                prefix=kohya_prefix,
                dtype=torch.float32,
            )
        )

    return kohya_state_dict


def save_sd_kohya_checkpoint(checkpoint_path: Path, unet: peft.PeftModel | None, text_encoder: peft.PeftModel | None):
    kohya_prefixes = []
    models = []
    for kohya_prefix, peft_model in zip([SD_KOHYA_UNET_KEY, SD_KOHYA_TEXT_ENCODER_KEY], [unet, text_encoder]):
        if peft_model is not None:
            kohya_prefixes.append(kohya_prefix)
            models.append(peft_model)

    kohya_state_dict = _convert_peft_models_to_kohya_state_dict(kohya_prefixes=kohya_prefixes, models=models)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_state_dict(kohya_state_dict, checkpoint_path)


def save_sdxl_kohya_checkpoint(
    checkpoint_path: Path,
    unet: peft.PeftModel | None,
    text_encoder_1: peft.PeftModel | None,
    text_encoder_2: peft.PeftModel | None,
):
    kohya_prefixes = []
    models = []
    for kohya_prefix, peft_model in zip(
        [SDXL_KOHYA_UNET_KEY, SDXL_KOHYA_TEXT_ENCODER_1_KEY, SDXL_KOHYA_TEXT_ENCODER_2_KEY],
        [unet, text_encoder_1, text_encoder_2],
    ):
        if peft_model is not None:
            kohya_prefixes.append(kohya_prefix)
            models.append(peft_model)

    kohya_state_dict = _convert_peft_models_to_kohya_state_dict(kohya_prefixes=kohya_prefixes, models=models)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_state_dict(kohya_state_dict, checkpoint_path)


def convert_sd_peft_checkpoint_to_kohya_state_dict(
    in_checkpoint_dir: Path,
    out_checkpoint_file: Path,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Convert SD v1 or SDXL PEFT models to a Kohya-format LoRA state dict."""
    # Get the immediate subdirectories of the checkpoint directory. We assume that each subdirectory is a PEFT model.
    peft_model_dirs = os.listdir(in_checkpoint_dir)
    peft_model_dirs = [in_checkpoint_dir / d for d in peft_model_dirs]  # Convert to Path objects.
    peft_model_dirs = [d for d in peft_model_dirs if d.is_dir()]  # Filter out non-directories.

    if len(peft_model_dirs) == 0:
        raise ValueError(f"No checkpoint files found in directory '{in_checkpoint_dir}'.")

    kohya_state_dict = {}
    for peft_model_dir in peft_model_dirs:
        if peft_model_dir.name in SD_PEFT_TO_KOHYA_KEYS:
            kohya_prefix = SD_PEFT_TO_KOHYA_KEYS[peft_model_dir.name]
        elif peft_model_dir.name in SDXL_PEFT_TO_KOHYA_KEYS:
            kohya_prefix = SDXL_PEFT_TO_KOHYA_KEYS[peft_model_dir.name]
        else:
            raise ValueError(f"Unrecognized checkpoint directory: '{peft_model_dir}'.")

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

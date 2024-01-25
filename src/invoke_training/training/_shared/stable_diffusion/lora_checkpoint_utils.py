from pathlib import Path

import peft
import torch
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel

from invoke_training.training._shared.checkpoints.serialization import save_state_dict

# Copied from https://github.com/huggingface/peft/blob/8665e2b5719faa4e4b91749ddec09442927b53e0/examples/stable_diffusion/train_dreambooth.py#L49C1-L65C87
# TODO(ryand): Is this the set of modules that we want to use?
# UNET_TARGET_MODULES = [
#     "to_q",
#     "to_k",
#     "to_v",
#     "proj",
#     "proj_in",
#     "proj_out",
#     "conv",
#     "conv1",
#     "conv2",
#     "conv_shortcut",
#     "to_out.0",
#     "time_emb_proj",
#     "ff.net.2",
# ]
# TEXT_ENCODER_TARGET_MODULES = ["fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj"]
# Module lists copied from diffusers training script:
UNET_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]
TEXT_ENCODER_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

SD_PEFT_UNET_KEY = "unet"
SD_PEFT_TEXT_ENCODER_KEY = "text_encoder"

SDXL_PEFT_UNET_KEY = "unet"
SDXL_PEFT_TEXT_ENCODER_1_KEY = "text_encoder_1"
SDXL_PEFT_TEXT_ENCODER_2_KEY = "text_encoder_2"


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


def save_sd_peft_checkpoint(
    checkpoint_dir: Path | str, unet: peft.PeftModel | None, text_encoder: peft.PeftModel | None
):
    models = {}
    if unet is not None:
        models[SD_PEFT_UNET_KEY] = unet
    if text_encoder is not None:
        models[SD_PEFT_TEXT_ENCODER_KEY] = text_encoder

    save_multi_model_peft_checkpoint(checkpoint_dir=checkpoint_dir, models=models)


def load_sd_peft_checkpoint(
    checkpoint_dir: Path | str, unet: UNet2DConditionModel, text_encoder: CLIPTextModel, is_trainable: bool = False
):
    models = load_multi_model_peft_checkpoint(
        checkpoint_dir=checkpoint_dir,
        models={SD_PEFT_UNET_KEY: unet, SD_PEFT_TEXT_ENCODER_KEY: text_encoder},
        is_trainable=is_trainable,
        raise_if_subdir_missing=False,
    )

    return models[SD_PEFT_UNET_KEY], models[SD_PEFT_TEXT_ENCODER_KEY]


def save_sdxl_peft_checkpoint(
    checkpoint_dir: Path | str,
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
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(lora_config.lora_alpha).to(dtype)

    return kohya_ss_state_dict


def convert_sd_peft_checkpoint_to_kohya_state_dict(
    in_checkpoint_dir: Path,
    out_checkpoint_file: Path,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Convert SD v1 PEFT models to a Kohya-format LoRA state dict."""

    if not in_checkpoint_dir.exists():
        raise ValueError(f"'{in_checkpoint_dir}' does not exist.")

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

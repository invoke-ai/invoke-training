import os
from pathlib import Path

import peft
import torch
from diffusers import FluxTransformer2DModel
from transformers import CLIPTextModel

from invoke_training._shared.checkpoints.serialization import save_state_dict

FLUX_TRANSFORMER_TARGET_MODULES = [
# double blocks
"attn.add_k_proj",
"attn.add_q_proj",
"attn.add_v_proj",
"attn.to_add_out",
"attn.to_k",
"attn.to_q",
"attn.to_v",
"attn.to_out.0",
"ff.net.0.proj",
"ff.net.2.0",
"ff_context.net.0.proj",
"ff_context.net.2.0",
# single blocks
"attn.to_k",
"attn.to_q",
"attn.to_v",
"proj_out",
]

TEXT_ENCODER_TARGET_MODULES = ["fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj", ""]

# Module lists copied from diffusers training script.
# These module lists will produce lighter, less expressive, LoRA models than the non-light versions.
FLUX_TRANSFORMER_TARGET_MODULES_LIGHT = ["to_k", "to_q", "to_v", "to_out.0"]
FLUX_TEXT_ENCODER_TARGET_MODULES_LIGHT = ["q_proj", "k_proj", "v_proj", "out_proj"]

FLUX_PEFT_TRANSFORMER_KEY = "transformer"
FLUX_PEFT_TEXT_ENCODER_1_KEY = "text_encoder_1"
FLUX_PEFT_TEXT_ENCODER_2_KEY = "text_encoder_2"


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
    checkpoint_dir: Path | str, transformer: FluxTransformer2DModel, text_encoder_1: CLIPTextModel, text_encoder_2: CLIPTextModel, is_trainable: bool = False
):
    models = load_multi_model_peft_checkpoint(
        checkpoint_dir=checkpoint_dir,
        models={
            FLUX_PEFT_TRANSFORMER_KEY: transformer,
            FLUX_PEFT_TEXT_ENCODER_1_KEY: text_encoder_1,
            FLUX_PEFT_TEXT_ENCODER_2_KEY: text_encoder_2,
        },
        is_trainable=is_trainable,
        raise_if_subdir_missing=False,
    )

    return models[FLUX_PEFT_TRANSFORMER_KEY], models[FLUX_PEFT_TEXT_ENCODER_1_KEY], models[
        FLUX_PEFT_TEXT_ENCODER_2_KEY
    ]


def save_flux_peft_checkpoint(
    checkpoint_dir: Path | str,
    transformer: peft.PeftModel | None,
    text_encoder_1: peft.PeftModel | None,
    text_encoder_2: peft.PeftModel | None,
):
    models = {}
    if transformer is not None:
        models[FLUX_PEFT_TRANSFORMER_KEY] = transformer
    if text_encoder_1 is not None:
        models[FLUX_PEFT_TEXT_ENCODER_1_KEY] = text_encoder_1
    if text_encoder_2 is not None:
        models[FLUX_PEFT_TEXT_ENCODER_2_KEY] = text_encoder_2

    save_multi_model_peft_checkpoint(checkpoint_dir=checkpoint_dir, models=models)


def load_flux_peft_checkpoint(
    checkpoint_dir: Path | str,
    transformer: FluxTransformer2DModel,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModel,
    is_trainable: bool = False,
):
    models = load_multi_model_peft_checkpoint(
        checkpoint_dir=checkpoint_dir,
        models={
            FLUX_PEFT_TRANSFORMER_KEY: transformer,
            FLUX_PEFT_TEXT_ENCODER_1_KEY: text_encoder_1,
            FLUX_PEFT_TEXT_ENCODER_2_KEY: text_encoder_2,
        },
        is_trainable=is_trainable,
        raise_if_subdir_missing=False,
    )

    return models[FLUX_PEFT_TRANSFORMER_KEY], models[FLUX_PEFT_TEXT_ENCODER_1_KEY], models[
        FLUX_PEFT_TEXT_ENCODER_2_KEY
    ]


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
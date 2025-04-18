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
    
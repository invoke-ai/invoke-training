import os
from pathlib import Path

import peft
import torch


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

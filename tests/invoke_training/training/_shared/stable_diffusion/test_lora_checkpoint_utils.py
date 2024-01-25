from pathlib import Path

import pytest

from invoke_training.training._shared.stable_diffusion.lora_checkpoint_utils import (
    convert_sd_peft_checkpoint_to_kohya_state_dict,
)


def test_convert_sd_peft_checkpoint_to_kohya_state_dict_raise_on_empty_directory(tmp_path: Path):
    with pytest.raises(ValueError, match="No checkpoint files found in directory"):
        convert_sd_peft_checkpoint_to_kohya_state_dict(
            in_checkpoint_dir=tmp_path, out_checkpoint_file=tmp_path / "out.safetensors"
        )


def test_convert_sd_peft_checkpoint_to_kohya_state_dict_raise_on_unexpected_subdirectory(tmp_path: Path):
    subdirectory = tmp_path / "subdir"
    subdirectory.mkdir()

    with pytest.raises(ValueError, match=f"Unrecognized checkpoint directory: '{subdirectory}'."):
        convert_sd_peft_checkpoint_to_kohya_state_dict(
            in_checkpoint_dir=tmp_path, out_checkpoint_file=tmp_path / "out.safetensors"
        )

import os
import tempfile

import pytest
import torch

from invoke_training.training.shared.serialization import (
    load_state_dict,
    save_state_dict,
)


@pytest.mark.parametrize("file_name", ["state.ckpt", "state.pt", "state.safetensors"])
def test_state_dict_save_and_load_roundtrip(file_name):
    with tempfile.TemporaryDirectory() as dir_name:
        file_path = os.path.join(dir_name, file_name)

        in_state_dict = {"a": torch.Tensor([1.0, 2.0])}

        # Perform save-load roundtrip.
        save_state_dict(in_state_dict, file_path)
        out_state_dict = load_state_dict(file_path)

    assert len(in_state_dict) == len(out_state_dict)
    for key in in_state_dict:
        assert torch.equal(in_state_dict[key], out_state_dict[key])


def test_save_state_dict_bad_extension():
    """Test that save_state_dict(...) raises a ValueError if it receives an unsupported file extension."""
    with pytest.raises(ValueError):
        save_state_dict({}, "state.txt")


def test_load_state_dict_bad_extension():
    """Test that load_state_dict(...) raises a ValueError if it receives an unsupported file extension."""
    with pytest.raises(ValueError):
        load_state_dict("state.txt")

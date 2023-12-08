from pathlib import Path

import pytest
import torch

from invoke_training.training.shared.data.transforms.tensor_disk_cache import TensorDiskCache


def test_tensor_disk_cache_roundtrip(tmp_path: Path):
    """Test a TensorDiskCache cache roundtrip."""
    cache = TensorDiskCache(str(tmp_path))

    in_dict = {"test_tensor": torch.rand((1, 2, 3)), "test_tuple": (1, 2), "test_list": [3, 4], "test_scalar": 1}

    # Roundtrip
    cache.save(0, in_dict)
    out_dict = cache.load(0)

    assert set(in_dict.keys()) == set(out_dict.keys())
    torch.testing.assert_close(out_dict["test_tensor"], in_dict["test_tensor"])
    assert out_dict["test_tuple"] == in_dict["test_tuple"]
    assert out_dict["test_list"] == in_dict["test_list"]
    assert out_dict["test_scalar"] == in_dict["test_scalar"]


def test_tensor_disk_cache_fail_overwrite(tmp_path):
    """Test that an attempt to overwrite an existing TensorDiskCache cache entry raises a ValueError."""
    cache = TensorDiskCache(str(tmp_path))
    in_dict = {"test_tensor": torch.rand((1, 2, 3))}
    cache.save(0, in_dict)

    with pytest.raises(AssertionError):
        cache.save(0, in_dict)

from pathlib import Path

import pytest
import torch

from invoke_training.training.shared.datasets.tensor_disk_cache import TensorDiskCache


def test_tensor_disk_cache_roundtrip(tmp_path: Path):
    """Test a TensorDiskCache cache roundtrip."""
    cache = TensorDiskCache(str(tmp_path))

    in_dict = {"test_tensor": torch.rand((1, 2, 3))}

    # Roundtrip
    cache.save(0, in_dict)
    out_dict = cache.load(0)

    assert set(in_dict.keys()) == set(out_dict.keys())
    for key in in_dict.keys():
        torch.testing.assert_close(out_dict[key], in_dict[key])


def test_tensor_disk_cache_fail_overwrite(tmp_path):
    """Test that an attempt to overwrite an existing TensorDiskCache cache entry raises a ValueError."""
    cache = TensorDiskCache(str(tmp_path))
    in_dict = {"test_tensor": torch.rand((1, 2, 3))}
    cache.save(0, in_dict)

    with pytest.raises(AssertionError):
        cache.save(0, in_dict)


def test_tensor_disk_cache_len(tmp_path: Path):
    """Test the TensorDiskCache len() function."""
    cache = TensorDiskCache(str(tmp_path))

    assert len(cache) == 0

    in_dict = {"test_tensor": torch.rand((1, 2, 3))}
    cache.save(0, in_dict)

    assert len(cache) == 1

import unittest.mock

import torch

from invoke_training.training.shared.data.transforms.load_cache_transform import LoadCacheTransform


def test_load_cache_transform():
    cached_tensor = torch.Tensor([1.0, 2.0, 3.0])
    mock_cache = unittest.mock.MagicMock()
    mock_cache.load.return_value = {"cached_tensor": cached_tensor}

    tf = LoadCacheTransform(
        cache=mock_cache, cache_key_field="cache_key", cache_field_to_output_field={"cached_tensor": "output"}
    )

    in_example = {"cache_key": 1}

    out_example = tf(in_example)

    mock_cache.load.assert_called_once_with(1)
    assert out_example["output"] is cached_tensor

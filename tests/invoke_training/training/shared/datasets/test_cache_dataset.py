import unittest

import torch

from invoke_training.training.shared.datasets.cache_dataset import CacheDataset


def test_cache_dataset_len():
    mock_cache = unittest.mock.MagicMock()
    mock_cache.__len__.return_value = 5

    dataset = CacheDataset(mock_cache)

    assert len(dataset) == 5
    mock_cache.__len__.assert_called_once()


def test_cache_dataset_getitem():
    test_example = {"test_tensor": torch.rand((1, 2, 3))}

    mock_cache = unittest.mock.MagicMock()
    mock_cache.load.return_value = test_example

    dataset = CacheDataset(mock_cache)

    example = dataset[0]

    assert example == test_example
    mock_cache.load.assert_called_with(0)

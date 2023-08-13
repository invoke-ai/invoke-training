import unittest

import pytest
import torch

from invoke_training.training.shared.datasets.merged_dataset import MergedDataset


def test_merged_dataset_len():
    """Test MergedDataset len()."""
    mock_dataset = unittest.mock.MagicMock()
    mock_dataset.__len__.return_value = 5

    dataset = MergedDataset([mock_dataset])

    assert len(dataset) == 5


def test_merged_dataset_length_mismatch():
    """Test that attempting to initialize a MergedDataset with datasets of incompatible lengths raises an exception."""
    mock_dataset_1 = unittest.mock.MagicMock()
    mock_dataset_1.__len__.return_value = 5
    mock_dataset_2 = unittest.mock.MagicMock()
    mock_dataset_2.__len__.return_value = 6  # Length does not match.

    with pytest.raises(ValueError):
        _ = MergedDataset([mock_dataset_1, mock_dataset_2])


def test_merged_dataset_getitem():
    """Test MergedDataset getitem function."""
    mock_dataset_1 = unittest.mock.MagicMock()
    mock_dataset_1.__len__.return_value = 1
    test_example_1 = {"test_tensor_1": torch.rand((1, 2, 3))}
    mock_dataset_1.__getitem__.return_value = test_example_1

    mock_dataset_2 = unittest.mock.MagicMock()
    mock_dataset_2.__len__.return_value = 1
    test_example_2 = {"test_tensor_2": torch.rand((1, 2, 3))}
    mock_dataset_2.__getitem__.return_value = test_example_2

    dataset = MergedDataset([mock_dataset_1, mock_dataset_2])

    example = dataset[0]

    assert set(example.keys()) == set(test_example_1.keys()).union(test_example_2.keys())
    for key in test_example_1.keys():
        torch.testing.assert_close(example[key], test_example_1[key])
    for key in test_example_2.keys():
        torch.testing.assert_close(example[key], test_example_2[key])

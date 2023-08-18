import unittest.mock

from invoke_training.training.shared.data.datasets.transform_dataset import (
    TransformDataset,
)


def test_transform_dataset_len():
    """Test the TransformDataset len() function."""
    mock_dataset = unittest.mock.MagicMock()
    mock_dataset.__len__.return_value = 5

    dataset = TransformDataset(mock_dataset, [])

    assert len(dataset) == 5


def test_transform_dataset_getitem():
    """Test the TransformDataset __getitem__() function."""
    field1 = 1
    field2 = "2"
    base_example = {"field1": field1}

    mock_dataset = unittest.mock.MagicMock()
    mock_dataset.__getitem__.return_value = base_example

    def mock_transform(example):
        example["field2"] = field2
        return example

    dataset = TransformDataset(mock_dataset, [mock_transform])

    out_example = dataset[0]

    assert out_example["field1"] == field1
    assert out_example["field2"] == field2

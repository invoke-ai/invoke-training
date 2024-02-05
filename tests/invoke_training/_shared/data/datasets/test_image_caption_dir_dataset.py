from pathlib import Path

import PIL.Image
import pytest

from invoke_training._shared.data.datasets.image_caption_dir_dataset import ImageCaptionDirDataset

from ..image_dir_fixture import image_caption_dir  # noqa: F401


def test_image_caption_dir_dataset_len(image_caption_dir):  # noqa: F811
    dataset = ImageCaptionDirDataset(str(image_caption_dir))

    assert len(dataset) == 5


def test_image_caption_dir_dataset_getitem(image_caption_dir):  # noqa: F811
    dataset = ImageCaptionDirDataset(str(image_caption_dir))

    example = dataset[0]

    assert set(example.keys()) == {"image", "id", "caption"}
    assert isinstance(example["image"], PIL.Image.Image)
    assert example["image"].mode == "RGB"
    assert example["id"] == "0"
    assert example["caption"] == "caption 0"


def test_image_caption_dir_dataset_keep_in_memory(image_caption_dir):  # noqa: F811
    dataset = ImageCaptionDirDataset(str(image_caption_dir), keep_in_memory=True)

    example = dataset[0]

    assert set(example.keys()) == {"image", "id", "caption"}
    assert isinstance(example["image"], PIL.Image.Image)
    assert example["image"].mode == "RGB"
    assert example["id"] == "0"
    assert example["caption"] == "caption 0"


def test_image_caption_dir_dataset_get_image_dimensions(image_caption_dir):  # noqa: F811
    dataset = ImageCaptionDirDataset(str(image_caption_dir))

    image_dims = dataset.get_image_dimensions()

    assert len(image_dims) == len(dataset)


def test_image_caption_dir_dataset_missing_caption_file(tmp_path: Path):  # noqa: F811
    # Create a directory with an image but no caption file.
    with open(tmp_path / "0.jpg", "w"):
        pass

    with pytest.raises(Exception, match=r"The following expected caption files are missing: \['.*0.txt'\]"):
        ImageCaptionDirDataset(str(tmp_path))

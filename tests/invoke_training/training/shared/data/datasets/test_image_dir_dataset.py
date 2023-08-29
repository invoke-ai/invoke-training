import PIL.Image

from invoke_training.training.shared.data.datasets.image_dir_dataset import (
    ImageDirDataset,
)

from ..image_dir_fixture import image_dir  # noqa: F401


def test_image_dir_dataset_len(image_dir):  # noqa: F811
    dataset = ImageDirDataset(str(image_dir))

    assert len(dataset) == 5


def test_image_dir_dataset_getitem(image_dir):  # noqa: F811
    dataset = ImageDirDataset(str(image_dir))

    example = dataset[0]

    assert set(example.keys()) == {"image", "id"}
    assert isinstance(example["image"], PIL.Image.Image)
    assert example["image"].mode == "RGB"
    assert example["id"] == 0

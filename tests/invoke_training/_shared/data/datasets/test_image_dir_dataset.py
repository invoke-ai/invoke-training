import PIL.Image

from invoke_training._shared.data.datasets.image_dir_dataset import ImageDirDataset

from ..dataset_fixtures import image_dir  # noqa: F401


def test_image_dir_dataset_len(image_dir):  # noqa: F811
    dataset = ImageDirDataset(str(image_dir))

    assert len(dataset) == 5


def test_image_dir_dataset_getitem(image_dir):  # noqa: F811
    dataset = ImageDirDataset(str(image_dir))

    example = dataset[0]

    assert set(example.keys()) == {"image", "id"}
    assert isinstance(example["image"], PIL.Image.Image)
    assert example["image"].mode == "RGB"
    assert example["id"] == "0"


def test_image_dir_dataset_keep_in_memory(image_dir):  # noqa: F811
    dataset = ImageDirDataset(str(image_dir), keep_in_memory=True)

    example = dataset[0]

    assert set(example.keys()) == {"image", "id"}
    assert isinstance(example["image"], PIL.Image.Image)
    assert example["image"].mode == "RGB"
    assert example["id"] == "0"


def test_image_dir_dataset_get_image_dimensions(image_dir):  # noqa: F811
    dataset = ImageDirDataset(str(image_dir))

    image_dims = dataset.get_image_dimensions()

    assert len(image_dims) == len(dataset)

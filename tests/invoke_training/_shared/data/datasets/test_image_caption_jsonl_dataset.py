import PIL.Image

from invoke_training._shared.data.datasets.image_caption_jsonl_dataset import ImageCaptionJsonlDataset

from ..image_dir_fixture import image_caption_jsonl  # noqa: F401


def test_image_caption_jsonl_dataset_len(image_caption_jsonl):  # noqa: F811
    dataset = ImageCaptionJsonlDataset(str(image_caption_jsonl))

    assert len(dataset) == 5


def test_image_caption_jsonl_dataset_getitem(image_caption_jsonl):  # noqa: F811
    dataset = ImageCaptionJsonlDataset(str(image_caption_jsonl))

    example = dataset[0]

    assert set(example.keys()) == {"image", "id", "caption"}
    assert isinstance(example["image"], PIL.Image.Image)
    assert example["image"].mode == "RGB"
    assert example["id"] == "0"
    assert example["caption"] == "caption 0"


def test_image_caption_jsonl_dataset_get_image_dimensions(image_caption_jsonl):  # noqa: F811
    dataset = ImageCaptionJsonlDataset(str(image_caption_jsonl))

    image_dims = dataset.get_image_dimensions()

    assert len(image_dims) == len(dataset)

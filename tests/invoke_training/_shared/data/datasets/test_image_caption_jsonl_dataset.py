import PIL.Image

from invoke_training._shared.data.datasets.image_caption_jsonl_dataset import ImageCaptionJsonlDataset

from ..dataset_fixtures import image_caption_jsonl  # noqa: F401


def test_image_caption_jsonl_dataset_len(image_caption_jsonl):  # noqa: F811
    dataset = ImageCaptionJsonlDataset(str(image_caption_jsonl))

    assert len(dataset) == 5


def test_image_caption_jsonl_dataset_getitem(image_caption_jsonl):  # noqa: F811
    dataset = ImageCaptionJsonlDataset(str(image_caption_jsonl))

    example = dataset[0]

    assert set(example.keys()) == {"image", "id", "caption", "mask"}
    assert isinstance(example["image"], PIL.Image.Image)
    assert example["image"].mode == "RGB"
    assert example["id"] == "0"
    assert example["caption"] == "caption 0"
    assert isinstance(example["mask"], PIL.Image.Image)
    assert example["mask"].mode == "L"


def test_image_caption_jsonl_dataset_keep_in_memory(image_caption_jsonl):  # noqa: F811
    dataset = ImageCaptionJsonlDataset(str(image_caption_jsonl), keep_in_memory=True)

    example = dataset[0]

    assert set(example.keys()) == {"image", "id", "caption", "mask"}
    assert isinstance(example["image"], PIL.Image.Image)
    assert example["image"].mode == "RGB"
    assert example["id"] == "0"
    assert example["caption"] == "caption 0"
    assert isinstance(example["mask"], PIL.Image.Image)
    assert example["mask"].mode == "L"

    # Confirm that accessing the same example again returns a shallow copy of the original example.
    # In other words, modifying the returned dict should not modify the cached example, but the same image should be
    # returned.
    same_example = dataset[0]
    assert same_example is not example
    assert same_example["image"] is example["image"]


def test_image_caption_jsonl_dataset_get_image_dimensions(image_caption_jsonl):  # noqa: F811
    dataset = ImageCaptionJsonlDataset(str(image_caption_jsonl))

    image_dims = dataset.get_image_dimensions()

    assert len(image_dims) == len(dataset)

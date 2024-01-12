import json
from pathlib import Path

import numpy as np
import PIL
import pytest
from PIL import Image

from invoke_training.training.shared.data.datasets.hf_image_caption_dataset import (
    HFImageCaptionDataset,
)
from invoke_training.training.shared.data.utils.resolution import Resolution

################################################
# Tests for HFImageCaptionDataset.from_dir(...)
################################################


def create_hf_imagefolder_dataset(tmp_dir: Path, num_images: int):
    """Construct a mock Hugging Face imagefolder dataset in a temporary directory.

    Args:
        tmp_dir (Path): The temporary directory where the mock dataset will be created.
        num_images (int): The number of mock images to include in the dataset.
    """
    # Construct mock images and save them to disk.
    rel_img_paths = []
    for i in range(num_images):
        rgb_np = np.ones((128, 128, 3), dtype=np.uint8)
        rgb_pil = Image.fromarray(rgb_np)
        rel_img_path = f"{i}.jpg"
        rel_img_paths.append(rel_img_path)
        rgb_pil.save(tmp_dir / rel_img_path)

    # Construct a mock metadata dict.
    metadata = []
    for rel_img_path in rel_img_paths:
        metadata.append({"file_name": rel_img_path, "text": f"Caption for {rel_img_path}"})

    # Write the metadata.jsonl to disk.
    metadata_path = tmp_dir / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for metadata_line in metadata:
            f.write(json.dumps(metadata_line) + "\n")


@pytest.fixture(scope="session")
def hf_imagefolder_dir(tmp_path_factory: pytest.TempPathFactory):
    """A fixture that prepares a temp directory with a mock Hugging Face imagefolder dataset and returns the directory
    path.

    Note that the 'session' scope is used to share the same directory across all tests in a session, because it is
    costly to populate the directory.

    Refer to https://docs.pytest.org/en/7.4.x/how-to/tmp_path.html#the-tmp-path-factory-fixture for details on the use
    of tmp_path_factory.
    """
    tmp_dir = tmp_path_factory.mktemp("dataset")
    create_hf_imagefolder_dataset(tmp_dir, 5)
    return tmp_dir


@pytest.fixture()
def hf_dir_dataset(hf_imagefolder_dir: Path):
    return HFImageCaptionDataset.from_dir(str(hf_imagefolder_dir))


def test_hf_dir_image_caption_dataset_bad_image_column(hf_imagefolder_dir: Path):
    """Test that a ValueError is raised if HFImageCaptionDataset is initialized with an `image_column` that does not
    exist.
    """
    with pytest.raises(ValueError):
        _ = HFImageCaptionDataset.from_dir(str(hf_imagefolder_dir), image_column="does_not_exist")


def test_hf_dir_image_caption_dataset_bad_caption_column(hf_imagefolder_dir: Path):
    """Test that a ValueError is raised if HFImageCaptionDataset is initialized with a `caption_column` that does not
    exist.
    """
    with pytest.raises(ValueError):
        _ = HFImageCaptionDataset.from_dir(str(hf_imagefolder_dir), caption_column="does_not_exist")


def test_hf_dir_image_caption_dataset_len(hf_dir_dataset: HFImageCaptionDataset):
    """Test the behaviour of HFImageCaptionDataset.__len__()."""
    assert len(hf_dir_dataset) == 5


def test_hf_dir_image_caption_dataset_index_error(hf_dir_dataset: HFImageCaptionDataset):
    """Test that an IndexError is raised if a dataset element is accessed with an index that is out-of-bounds."""
    with pytest.raises(IndexError):
        _ = hf_dir_dataset[1000]


def test_hf_dir_image_caption_dataset_getitem(hf_dir_dataset: HFImageCaptionDataset):
    """Test that HFImageCaptionDataset.__getitem__(...) returns a valid example."""
    example = hf_dir_dataset[0]

    assert set(example.keys()) == {"image", "caption", "id"}
    assert isinstance(example["image"], PIL.Image.Image)
    assert example["image"].mode == "RGB"
    assert isinstance(example["caption"], str)
    assert example["id"] == 0


def test_hf_dir_image_caption_dataset_get_image_dimensions(hf_dir_dataset: HFImageCaptionDataset):
    """Test HFImageCaptionDataset.get_image_dimensions()."""

    image_dims = hf_dir_dataset.get_image_dimensions()

    assert len(image_dims) == 5
    for image_dim in image_dims:
        assert image_dim == Resolution(128, 128)


################################################
# Tests for HFImageCaptionDataset.from_hub(...)
################################################


@pytest.mark.loads_model
def test_hf_hub_image_caption_dataset_bad_image_column():
    """Test that a ValueError is raised if HFImageCaptionDataset is initialized with an `image_column` that does not
    exist.
    """
    with pytest.raises(ValueError):
        _ = HFImageCaptionDataset.from_hub(
            "lambdalabs/pokemon-blip-captions",
            hf_load_dataset_kwargs={"revision": "8b762e1dac1b31d60e01ee8f08a9d8a232b59e17"},
            image_column="does_not_exist",
        )


@pytest.mark.loads_model
def test_hf_hub_image_caption_dataset_bad_caption_column():
    """Test that a ValueError is raised if HFImageCaptionDataset is initialized with a `caption_column` that does not
    exist.
    """
    with pytest.raises(ValueError):
        _ = HFImageCaptionDataset.from_hub(
            "lambdalabs/pokemon-blip-captions",
            hf_load_dataset_kwargs={"revision": "8b762e1dac1b31d60e01ee8f08a9d8a232b59e17"},
            caption_column="does_not_exist",
        )


@pytest.fixture
def hf_hub_dataset():
    return HFImageCaptionDataset.from_hub(
        "lambdalabs/pokemon-blip-captions",
        hf_load_dataset_kwargs={"revision": "8b762e1dac1b31d60e01ee8f08a9d8a232b59e17"},
    )


@pytest.mark.loads_model
def test_hf_hub_image_caption_dataset_index_error(hf_hub_dataset: HFImageCaptionDataset):
    """Test that an IndexError is raised if a dataset element is accessed with an index that is out-of-bounds."""
    with pytest.raises(IndexError):
        _ = hf_hub_dataset[1000]


@pytest.mark.loads_model
def test_hf_hub_image_caption_dataset_len(hf_hub_dataset: HFImageCaptionDataset):
    """Test the behaviour of HFImageCaptionDataset.__len__()."""
    # Expected dataset length was checked manually here:
    # https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions
    assert len(hf_hub_dataset) == 833


@pytest.mark.loads_model
def test_hf_hub_image_caption_dataset_getitem(hf_hub_dataset: HFImageCaptionDataset):
    """Test that HFImageCaptionDataset.__getitem__(...) returns a valid example."""
    example = hf_hub_dataset[0]

    assert set(example.keys()) == {"image", "caption", "id"}
    assert isinstance(example["image"], PIL.Image.Image)
    assert example["image"].mode == "RGB"
    assert isinstance(example["caption"], str)
    assert example["id"] == 0


@pytest.mark.loads_model
def test_hf_hub_image_caption_dataset_get_image_dimensions(hf_hub_dataset: HFImageCaptionDataset):
    """Test HFImageCaptionDataset.get_image_dimensions()."""
    image_dims = hf_hub_dataset.get_image_dimensions()

    # This is just a smoke test. We don't actually check that the dimensions are correct.
    assert len(image_dims) == 833

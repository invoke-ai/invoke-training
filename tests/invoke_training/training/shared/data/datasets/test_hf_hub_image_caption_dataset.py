import PIL
import pytest

from invoke_training.training.shared.data.datasets.hf_hub_image_caption_dataset import HFHubImageCaptionDataset


@pytest.mark.loads_model
def test_hf_hub_image_caption_dataset_bad_image_column():
    """Test that a ValueError is raised if HFHubImageCaptionDataset is initialized with an `image_column` that does not
    exist.
    """
    with pytest.raises(ValueError):
        _ = HFHubImageCaptionDataset(
            "lambdalabs/pokemon-blip-captions",
            hf_load_dataset_kwargs={"revision": "8b762e1dac1b31d60e01ee8f08a9d8a232b59e17"},
            image_column="does_not_exist",
        )


@pytest.mark.loads_model
def test_hf_hub_image_caption_dataset_bad_caption_column():
    """Test that a ValueError is raised if HFHubImageCaptionDataset is initialized with a `caption_column` that does not
    exist.
    """
    with pytest.raises(ValueError):
        _ = HFHubImageCaptionDataset(
            "lambdalabs/pokemon-blip-captions",
            hf_load_dataset_kwargs={"revision": "8b762e1dac1b31d60e01ee8f08a9d8a232b59e17"},
            caption_column="does_not_exist",
        )


@pytest.fixture
def hf_hub_dataset():
    return HFHubImageCaptionDataset(
        "lambdalabs/pokemon-blip-captions",
        hf_load_dataset_kwargs={"revision": "8b762e1dac1b31d60e01ee8f08a9d8a232b59e17"},
    )


@pytest.mark.loads_model
def test_hf_hub_image_caption_dataset_len(hf_hub_dataset: HFHubImageCaptionDataset):
    """Test the behaviour of HFHubImageCaptionDataset.__len__()."""
    # Expected dataset length was checked manually here:
    # https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions
    assert len(hf_hub_dataset) == 833


@pytest.mark.loads_model
def test_hf_hub_image_caption_dataset_index_error(hf_hub_dataset: HFHubImageCaptionDataset):
    """Test that an IndexError is raised if a dataset element is accessed with an index that is out-of-bounds."""
    with pytest.raises(IndexError):
        _ = hf_hub_dataset[1000]


@pytest.mark.loads_model
def test_hf_hub_image_caption_dataset_getitem(hf_hub_dataset: HFHubImageCaptionDataset):
    """Test that HFHubImageCaptionDataset.__getitem__(...) returns a valid example."""
    example = hf_hub_dataset[0]

    assert set(example.keys()) == {"image", "caption", "id"}
    assert isinstance(example["image"], PIL.Image.Image)
    assert example["image"].mode == "RGB"
    assert isinstance(example["caption"], str)
    assert example["id"] == 0


@pytest.mark.loads_model
def test_hf_hub_image_caption_dataset_get_image_dimensions(hf_hub_dataset: HFHubImageCaptionDataset):
    """Test HFHubImageCaptionDataset.get_image_dimensions()."""

    image_dims = hf_hub_dataset.get_image_dimensions()

    # This is just a smoke test. We don't currently check that the dimensions are correct.
    assert len(image_dims) == 833

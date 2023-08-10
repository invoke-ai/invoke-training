import PIL
import pytest

from invoke_training.training.shared.datasets.hf_hub_image_caption_reader import (
    HFHubImageCaptionReader,
)


@pytest.mark.loads_model
def test_hf_hub_image_caption_reader_bad_image_column():
    """Test that a ValueError is raised if HFHubImageCaptionReader is initialized with an `image_column` that does not
    exist.
    """
    with pytest.raises(ValueError):
        _ = HFHubImageCaptionReader(
            "lambdalabs/pokemon-blip-captions",
            hf_load_dataset_kwargs={"revision": "8b762e1dac1b31d60e01ee8f08a9d8a232b59e17"},
            image_column="does_not_exist",
        )


@pytest.mark.loads_model
def test_hf_hub_image_caption_reader_bad_caption_column():
    """Test that a ValueError is raised if HFHubImageCaptionReader is initialized with a `caption_column` that does not
    exist.
    """
    with pytest.raises(ValueError):
        _ = HFHubImageCaptionReader(
            "lambdalabs/pokemon-blip-captions",
            hf_load_dataset_kwargs={"revision": "8b762e1dac1b31d60e01ee8f08a9d8a232b59e17"},
            caption_column="does_not_exist",
        )


@pytest.fixture
def hf_hub_dataset():
    return HFHubImageCaptionReader(
        "lambdalabs/pokemon-blip-captions",
        hf_load_dataset_kwargs={"revision": "8b762e1dac1b31d60e01ee8f08a9d8a232b59e17"},
    )


@pytest.mark.loads_model
def test_hf_hub_image_caption_reader_len(hf_hub_dataset: HFHubImageCaptionReader):
    """Test the behaviour of HFHubImageCaptionReader.__len__()."""
    # Expected dataset length was checked manually here:
    # https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions
    assert len(hf_hub_dataset) == 833


@pytest.mark.loads_model
def test_hf_hub_image_caption_reader_index_error(hf_hub_dataset: HFHubImageCaptionReader):
    """Test that an IndexError is raised if a dataset element is accessed with an index that is out-of-bounds."""
    with pytest.raises(IndexError):
        _ = hf_hub_dataset[1000]


@pytest.mark.loads_model
def test_hf_hub_image_caption_reader_getitem(hf_hub_dataset: HFHubImageCaptionReader):
    """Test that HFHubImageCaptionReader.__getitem__(...) returns a valid example."""
    example = hf_hub_dataset[0]

    assert set(example.keys()) == {"image", "caption"}
    assert isinstance(example["image"], PIL.Image.Image)
    assert example["image"].mode == "RGB"
    assert isinstance(example["caption"], str)

import pytest
from datasets import VerificationMode
from PIL.Image import Image

from invoke_training.training._shared.data.datasets.hf_image_pair_preference_dataset import HFImagePairPreferenceDataset


@pytest.mark.loads_model
def test_hf_hub_image_caption_dataset_getitem():
    """Test that HFImagePairPreferenceDataset.__getitem__(...) returns a valid example."""
    # HACK(ryand): This funky configuration is done so that we just download a small slice of the very large
    # 'yuvalkirstain/pickapic_v2' dataset.
    dataset = HFImagePairPreferenceDataset.from_hub(
        "yuvalkirstain/pickapic_v2",
        split="validation_unique",
        hf_load_dataset_kwargs={
            "data_files": {
                "validation_unique": "data/validation_unique-00000-of-00001-33ead111845fc9c4.parquet",
            },
            # Disable checks so that it doesn't complain that I haven't downloaded the other splits.
            "verification_mode": VerificationMode.NO_CHECKS,
        },
    )

    example = dataset[0]

    assert set(example.keys()) == {"id", "image_0", "image_1", "prefer_0", "prefer_1", "caption"}

    assert example["id"] == 0

    assert isinstance(example["image_0"], Image)
    assert example["image_0"].mode == "RGB"
    assert isinstance(example["image_1"], Image)
    assert example["image_1"].mode == "RGB"

    assert isinstance(example["prefer_0"], bool)
    assert isinstance(example["prefer_1"], bool)
    # The following is not always true, but is usually true.
    assert example["prefer_0"] != example["prefer_1"]

    assert isinstance(example["caption"], str)


@pytest.mark.loads_model
def test_hf_hub_image_caption_dataset_len():
    """Test that HFImagePairPreferenceDataset.__len__(...) returns the correct value."""
    # HACK(ryand): This funky configuration is done so that we just download a small slice of the very large
    # 'yuvalkirstain/pickapic_v2' dataset.
    dataset = HFImagePairPreferenceDataset.from_hub(
        "yuvalkirstain/pickapic_v2",
        skip_no_preference=False,
        split="validation_unique",
        hf_load_dataset_kwargs={
            "data_files": {
                "validation_unique": "data/validation_unique-00000-of-00001-33ead111845fc9c4.parquet",
            },
            # Disable checks so that it doesn't complain that I haven't downloaded the other splits.
            "verification_mode": VerificationMode.NO_CHECKS,
        },
    )

    assert len(dataset) == 500


@pytest.mark.loads_model
def test_hf_hub_image_caption_dataset_skip_no_preference_len():
    """Test the HFImagePairPreferenceDataset skip_no_preference parameter."""
    # HACK(ryand): This funky configuration is done so that we just download a small slice of the very large
    # 'yuvalkirstain/pickapic_v2' dataset.
    dataset = HFImagePairPreferenceDataset.from_hub(
        "yuvalkirstain/pickapic_v2",
        skip_no_preference=True,
        split="validation_unique",
        hf_load_dataset_kwargs={
            "data_files": {
                "validation_unique": "data/validation_unique-00000-of-00001-33ead111845fc9c4.parquet",
            },
            # Disable checks so that it doesn't complain that I haven't downloaded the other splits.
            "verification_mode": VerificationMode.NO_CHECKS,
        },
    )

    assert len(dataset) == 429

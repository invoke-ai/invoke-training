from datasets import VerificationMode

from invoke_training.config.shared.data.dataset_config import (
    HFDirImageCaptionDatasetConfig,
    HFHubImageCaptionDatasetConfig,
    HFHubImagePairPreferenceDatasetConfig,
)
from invoke_training.training.shared.data.datasets.hf_image_caption_dataset import HFImageCaptionDataset
from invoke_training.training.shared.data.datasets.hf_image_pair_preference_dataset import HFImagePairPreferenceDataset


def build_hf_hub_image_caption_dataset(config: HFHubImageCaptionDatasetConfig) -> HFImageCaptionDataset:
    return HFImageCaptionDataset.from_hub(
        dataset_name=config.dataset_name,
        hf_load_dataset_kwargs={
            "name": config.dataset_config_name,
            "cache_dir": config.hf_cache_dir,
        },
        image_column=config.image_column,
        caption_column=config.caption_column,
    )


def build_hf_dir_image_caption_dataset(config: HFDirImageCaptionDatasetConfig) -> HFImageCaptionDataset:
    return HFImageCaptionDataset.from_dir(
        dataset_dir=config.dataset_dir,
        hf_load_dataset_kwargs=None,
        image_column=config.image_column,
        caption_column=config.caption_column,
    )


def build_hf_image_pair_preference_dataset(
    config: HFHubImagePairPreferenceDatasetConfig,
) -> HFImagePairPreferenceDataset:
    # HACK(ryand): This is currently hard-coded to just download a small slice of the very large
    # 'yuvalkirstain/pickapic_v2' dataset.
    return HFImagePairPreferenceDataset.from_hub(
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

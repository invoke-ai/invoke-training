from datasets import VerificationMode

from invoke_training._shared.data.datasets.hf_image_caption_dataset import HFImageCaptionDataset
from invoke_training._shared.data.datasets.hf_image_pair_preference_dataset import HFImagePairPreferenceDataset
from invoke_training._shared.data.datasets.image_caption_dir_dataset import ImageCaptionDirDataset
from invoke_training._shared.data.datasets.image_caption_jsonl_dataset import ImageCaptionJsonlDataset
from invoke_training.config.data.dataset_config import (
    HFHubImageCaptionDatasetConfig,
    ImageCaptionDirDatasetConfig,
    ImageCaptionJsonlDatasetConfig,
)
from invoke_training.pipelines._experimental.sd_dpo_lora.config import HFHubImagePairPreferenceDatasetConfig


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


def build_image_caption_jsonl_dataset(config: ImageCaptionJsonlDatasetConfig) -> HFImageCaptionDataset:
    return ImageCaptionJsonlDataset(
        jsonl_path=config.jsonl_path,
        image_column=config.image_column,
        caption_column=config.caption_column,
        keep_in_memory=config.keep_in_memory,
    )


def build_image_caption_dir_dataset(config: ImageCaptionDirDatasetConfig) -> ImageCaptionDirDataset:
    return ImageCaptionDirDataset(
        dataset_dir=config.dataset_dir,
        keep_in_memory=config.keep_in_memory,
    )


def build_hf_image_pair_preference_dataset(
    config: HFHubImagePairPreferenceDatasetConfig,
) -> HFImagePairPreferenceDataset:
    # HACK(ryand): This is currently hard-coded to just download a small slice of the very large
    # 'yuvalkirstain/pickapic_v2' dataset.
    return HFImagePairPreferenceDataset.from_hub(
        "yuvalkirstain/pickapic_v2",
        split="train",
        hf_load_dataset_kwargs={
            "data_files": {
                # "validation_unique": "data/validation_unique-00000-of-00001-33ead111845fc9c4.parquet",
                "train": [
                    "data/train-00000-of-00645-b66ac786bf6fb553.parquet",
                    "data/train-00001-of-00645-c7b349dd222d6515.parquet",
                    "data/train-00002-of-00645-e4f54d615a978deb.parquet",
                    "data/train-00003-of-00645-2b9d59bac8b433ff.parquet",
                    "data/train-00004-of-00645-e4964649dc0ea543.parquet",
                    "data/train-00005-of-00645-45e8efc0fe93f6e9.parquet",
                ]
            },
            # Disable checks so that it doesn't complain that I haven't downloaded the other splits.
            "verification_mode": VerificationMode.NO_CHECKS,
        },
    )

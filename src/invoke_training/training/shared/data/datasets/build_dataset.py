from invoke_training.config.shared.data.dataset_config import (
    HFDirImageCaptionDatasetConfig,
    HFHubImageCaptionDatasetConfig,
)
from invoke_training.training.shared.data.datasets.hf_image_caption_dataset import HFImageCaptionDataset


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

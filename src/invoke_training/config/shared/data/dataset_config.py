from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class HFHubImageCaptionDatasetConfig(BaseModel):
    type: Literal["HF_HUB_IMAGE_CAPTION_DATASET"] = "HF_HUB_IMAGE_CAPTION_DATASET"

    # The name of a Hugging Face dataset.
    dataset_name: str

    # The Hugging Face dataset config name. Leave as None if there's only one config.
    dataset_config_name: Optional[str] = None

    # The Hugging Face cache directory to use for dataset downloads.
    # If None, the default value will be used (usually '~/.cache/huggingface/datasets').
    hf_cache_dir: Optional[str] = None

    # The name of the dataset column that contains image paths.
    image_column: str = "image"

    # The name of the dataset column that contains captions.
    caption_column: str = "text"


class HFDirImageCaptionDatasetConfig(BaseModel):
    type: Literal["HF_DIR_IMAGE_CAPTION_DATASET"] = "HF_DIR_IMAGE_CAPTION_DATASET"

    # The directory to load a dataset from. The dataset is expected to be in
    # Hugging Face imagefolder format (https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder).
    dataset_dir: str

    # The name of the dataset column that contains image paths.
    image_column: str = "image"

    # The name of the dataset column that contains captions.
    caption_column: str = "text"


class ImageDirDatasetConfig(BaseModel):
    type: Literal["IMAGE_DIR_DATASET"] = "IMAGE_DIR_DATASET"

    # The directory to load images from.
    dataset_dir: str


# Datasets that produce image-caption pairs.
ImageCaptionDatasetConfig = Annotated[
    Union[HFDirImageCaptionDatasetConfig, HFHubImageCaptionDatasetConfig], Field(discriminator="type")
]

# All datasets.
DatasetConfig = Annotated[
    Union[HFDirImageCaptionDatasetConfig, HFHubImageCaptionDatasetConfig, ImageDirDatasetConfig],
    Field(discriminator="type"),
]

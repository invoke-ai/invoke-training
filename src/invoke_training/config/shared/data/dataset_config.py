from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class HFHubImageCaptionDatasetConfig(BaseModel):
    type: Literal["HF_HUB_IMAGE_CAPTION_DATASET"] = "HF_HUB_IMAGE_CAPTION_DATASET"

    dataset_name: str
    """The name of a Hugging Face dataset.
    """

    dataset_config_name: Optional[str] = None
    """The Hugging Face dataset config name. Leave as None if there's only one config.
    """

    hf_cache_dir: Optional[str] = None
    """The Hugging Face cache directory to use for dataset downloads.
    If None, the default value will be used (usually '~/.cache/huggingface/datasets').
    """

    image_column: str = "image"
    """The name of the dataset column that contains image paths.
    """

    caption_column: str = "text"
    """The name of the dataset column that contains captions.
    """


class HFDirImageCaptionDatasetConfig(BaseModel):
    type: Literal["HF_DIR_IMAGE_CAPTION_DATASET"] = "HF_DIR_IMAGE_CAPTION_DATASET"

    dataset_dir: str
    """The directory to load a dataset from. The dataset is expected to be in
    Hugging Face imagefolder format (https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder)."""

    image_column: str = "image"
    """The name of the dataset column that contains image paths.
    """

    caption_column: str = "text"
    """The name of the dataset column that contains captions.
    """


class ImageDirDatasetConfig(BaseModel):
    type: Literal["IMAGE_DIR_DATASET"] = "IMAGE_DIR_DATASET"

    dataset_dir: str
    """The directory to load images from."""


class HFHubImagePairPreferenceDatasetConfig(BaseModel):
    type: Literal["HF_HUB_IMAGE_PAIR_PREFERENCE_DATASET"] = "HF_HUB_IMAGE_PAIR_PREFERENCE_DATASET"

    # TODO(ryand): Fill this out.


class ImagePairPreferenceDatasetConfig(BaseModel):
    type: Literal["IMAGE_PAIR_PREFERENCE_DATASET"] = "IMAGE_PAIR_PREFERENCE_DATASET"

    dataset_dir: str
    """The directory to load the dataset from."""


# Datasets that produce image-caption pairs.
ImageCaptionDatasetConfig = Annotated[
    Union[HFDirImageCaptionDatasetConfig, HFHubImageCaptionDatasetConfig], Field(discriminator="type")
]

from typing import Annotated, Literal, Optional, Union

from pydantic import Field

from invoke_training.config.config_base_model import ConfigBaseModel


class HFHubImageCaptionDatasetConfig(ConfigBaseModel):
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


class ImageCaptionJsonlDatasetConfig(ConfigBaseModel):
    type: Literal["IMAGE_CAPTION_JSONL_DATASET"] = "IMAGE_CAPTION_JSONL_DATASET"

    jsonl_path: str
    """The path to a JSONL file containing image paths and captions."""

    image_column: str = "image"
    """The name of the dataset column that contains image paths.
    """

    caption_column: str = "text"
    """The name of the dataset column that contains captions.
    """


class ImageDirDatasetConfig(ConfigBaseModel):
    type: Literal["IMAGE_DIR_DATASET"] = "IMAGE_DIR_DATASET"

    dataset_dir: str
    """The directory to load images from."""

    keep_in_memory: bool = False
    """If `True`, load all images into memory on initialization so that they can be accessed quickly. If `False`, images
    are loaded from disk each time they are accessed. Setting to `True` improves performance for datasets that are small
    enough to be kept in memory.
    """


class ImageCaptionDirDatasetConfig(ConfigBaseModel):
    type: Literal["IMAGE_CAPTION_DIR_DATASET"] = "IMAGE_CAPTION_DIR_DATASET"

    dataset_dir: str
    """The directory to load images from."""

    keep_in_memory: bool = False
    """If `True`, load all images into memory on initialization so that they can be accessed quickly. If `False`, images
    are loaded from disk each time they are accessed. Setting to `True` improves performance for datasets that are small
    enough to be kept in memory.
    """


# Datasets that produce image-caption pairs.
ImageCaptionDatasetConfig = Annotated[
    Union[HFHubImageCaptionDatasetConfig, ImageCaptionJsonlDatasetConfig, ImageCaptionDirDatasetConfig],
    Field(discriminator="type"),
]

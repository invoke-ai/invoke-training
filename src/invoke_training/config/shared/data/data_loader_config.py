from typing import Literal, Optional

from pydantic import BaseModel

from invoke_training.config.shared.data.dataset_config import ImageCaptionDatasetConfig, ImageDirDatasetConfig
from invoke_training.config.shared.data.transform_config import (
    SDImageTransformConfig,
    ShuffleCaptionTransformConfig,
    TextualInversionCaptionConfig,
)


class ImageCaptionSDDataLoaderConfig(BaseModel):
    type: Literal["IMAGE_CAPTION_SD_DATA_LOADER"] = "IMAGE_CAPTION_SD_DATA_LOADER"

    dataset: ImageCaptionDatasetConfig

    image_transforms: SDImageTransformConfig

    dataloader_num_workers: int = 0
    """Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    """


class DreamboothSDDataLoaderConfig(BaseModel):
    type: Literal["DREAMBOOTH_SD_DATA_LOADER"] = "DREAMBOOTH_SD_DATA_LOADER"

    instance_caption: str
    class_caption: Optional[str] = None

    instance_dataset: ImageDirDatasetConfig
    class_dataset: Optional[ImageDirDatasetConfig] = None

    class_data_loss_weight: float = 1.0
    """The loss weight applied to class dataset examples. Instance dataset examples have an implicit loss weight of 1.0.
    """

    image_transforms: SDImageTransformConfig

    dataloader_num_workers: int = 0
    """Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    """


class DreamboothSDXLDataLoaderConfig(DreamboothSDDataLoaderConfig):
    type: Literal["DREAMBOOTH_SDXL_DATA_LOADER"] = "DREAMBOOTH_SDXL_DATA_LOADER"


class TextualInversionSDDataLoaderConfig(BaseModel):
    type: Literal["TEXTUAL_INVERSION_SD_DATA_LOADER"] = "TEXTUAL_INVERSION_SD_DATA_LOADER"

    dataset: ImageDirDatasetConfig

    captions: TextualInversionCaptionConfig

    image_transforms: SDImageTransformConfig
    """The image transforms to apply to all images.
    """

    shuffle_caption_transform: Optional[ShuffleCaptionTransformConfig] = None
    """The caption shuffling configuration. If None, caption shuffling is disabled.
    """

    dataloader_num_workers: int = 0
    """Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    """


class TextualInversionSDXLDataLoaderConfig(TextualInversionSDDataLoaderConfig):
    type: Literal["TEXTUAL_INVERSION_SDXL_DATA_LOADER"] = "TEXTUAL_INVERSION_SDXL_DATA_LOADER"

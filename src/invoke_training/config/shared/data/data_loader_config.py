from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

from invoke_training.config.shared.data.dataset_config import ImageCaptionDatasetConfig, ImageDirDatasetConfig
from invoke_training.config.shared.data.transform_config import (
    SDImageTransformConfig,
    SDXLImageTransformConfig,
    ShuffleCaptionTransformConfig,
    TextualInversionCaptionConfig,
)


class ImageCaptionSDDataLoaderConfig(BaseModel):
    type: Literal["IMAGE_CAPTION_SD_DATA_LOADER"] = "IMAGE_CAPTION_SD_DATA_LOADER"

    dataset: ImageCaptionDatasetConfig

    image_transforms: SDImageTransformConfig

    # Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    dataloader_num_workers: int = 0


class ImageCaptionSDXLDataLoaderConfig(ImageCaptionSDDataLoaderConfig):
    type: Literal["IMAGE_CAPTION_SDXL_DATA_LOADER"] = "IMAGE_CAPTION_SDXL_DATA_LOADER"

    image_transforms: SDXLImageTransformConfig


class DreamboothSDDataLoaderConfig(BaseModel):
    type: Literal["DREAMBOOTH_SD_DATA_LOADER"] = "DREAMBOOTH_SD_DATA_LOADER"

    instance_caption: str
    class_caption: Optional[str] = None

    instance_dataset: ImageDirDatasetConfig
    class_dataset: Optional[ImageDirDatasetConfig] = None

    # The loss weight applied to class dataset examples. Instance dataset examples have an implicit loss weight of 1.0.
    class_data_loss_weight: float = 1.0

    image_transforms: SDImageTransformConfig

    # Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    dataloader_num_workers: int = 0


class DreamboothSDXLDataLoaderConfig(DreamboothSDDataLoaderConfig):
    type: Literal["DREAMBOOTH_SDXL_DATA_LOADER"] = "DREAMBOOTH_SDXL_DATA_LOADER"

    image_transforms: SDXLImageTransformConfig


class TextualInversionSDDataLoaderConfig(BaseModel):
    type: Literal["TEXTUAL_INVERSION_SD_DATA_LOADER"] = "TEXTUAL_INVERSION_SD_DATA_LOADER"

    dataset: ImageDirDatasetConfig

    captions: TextualInversionCaptionConfig

    # The image transforms to apply to all images.
    image_transforms: SDImageTransformConfig

    # The caption shuffling configuration. If None, caption shuffling is disabled.
    shuffle_caption_transform: Optional[ShuffleCaptionTransformConfig] = None

    # Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    dataloader_num_workers: int = 0


# All data loaders.
DataLoaderConfig = Annotated[
    Union[
        ImageCaptionSDDataLoaderConfig,
        ImageCaptionSDXLDataLoaderConfig,
        DreamboothSDDataLoaderConfig,
        DreamboothSDXLDataLoaderConfig,
        TextualInversionSDDataLoaderConfig,
    ],
    Field(discriminator="type"),
]

from typing import Literal, Optional

from invoke_training.config.shared.config_base_model import ConfigBaseModel
from invoke_training.config.shared.data.dataset_config import (
    ImageCaptionDatasetConfig,
    ImageDirDatasetConfig,
)
from invoke_training.config.shared.data.transform_config import (
    ShuffleCaptionTransformConfig,
    TextualInversionCaptionConfig,
)


class AspectRatioBucketConfig(ConfigBaseModel):
    target_resolution: int
    """The target resolution for all aspect ratios. When generating aspect ratio buckets, the resolution of each bucket
    is selected to have roughly `target_resolution * target_resolution` pixels (i.e. a square image with dimensions
    equal to `target_resolution`).
    """

    start_dim: int
    """Aspect ratio bucket resolutions are generated as follows:

    - Iterate over 'first' dimension values from `start_dim` to `end_dim` in steps of size `divisible_by`.
    - Calculate the 'second' dimension to be as close as possible to the total number of pixels in `target_resolution`,
    while still being divisible by `divisible_by`.

    tip: Choosing aspect ratio buckets
        The aspect ratio bucket resolutions are logged at the start of training with the number of images in each
        bucket. Review these logs to make sure that images are being split into buckets as expected.

        Highly fragmented splits (i.e. many buckets with few examples in each) can 1) limit the extent to which examples
        can be shuffled, and 2) slow down training if there are many partial batches.
    """
    end_dim: int
    """See explanation under
    [`start_dim`][invoke_training.config.shared.data.data_loader_config.AspectRatioBucketConfig.start_dim].
    """

    divisible_by: int
    """See explanation under
    [`start_dim`][invoke_training.config.shared.data.data_loader_config.AspectRatioBucketConfig.start_dim].
    """


class ImageCaptionSDDataLoaderConfig(ConfigBaseModel):
    type: Literal["IMAGE_CAPTION_SD_DATA_LOADER"] = "IMAGE_CAPTION_SD_DATA_LOADER"

    dataset: ImageCaptionDatasetConfig

    aspect_ratio_buckets: AspectRatioBucketConfig | None = None

    resolution: int | tuple[int, int] = 512
    """The resolution for input images. Either a scalar integer representing the square resolution height and width, or
    a (height, width) tuple. All of the images in the dataset will be resized to this resolution unless the
    `aspect_ratio_buckets` config is set.
    """

    center_crop: bool = True
    """If True, input images will be center-cropped to the target resolution.
    If False, input images will be randomly cropped to the target resolution.
    """

    random_flip: bool = False
    """Whether random flip augmentations should be applied to input images.
    """

    caption_prefix: str | None = None
    """A prefix that will be prepended to all captions. If None, no prefix will be added.
    """

    dataloader_num_workers: int = 0
    """Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    """


class DreamboothSDDataLoaderConfig(ConfigBaseModel):
    type: Literal["DREAMBOOTH_SD_DATA_LOADER"] = "DREAMBOOTH_SD_DATA_LOADER"

    instance_caption: str
    class_caption: Optional[str] = None

    instance_dataset: ImageDirDatasetConfig
    class_dataset: Optional[ImageDirDatasetConfig] = None

    class_data_loss_weight: float = 1.0
    """The loss weight applied to class dataset examples. Instance dataset examples have an implicit loss weight of 1.0.
    """

    aspect_ratio_buckets: AspectRatioBucketConfig | None = None
    """The aspect ratio bucketing configuration. If None, aspect ratio bucketing is disabled, and all images will be
    resized to the same resolution.
    """

    resolution: int | tuple[int, int] = 512
    """The resolution for input images. Either a scalar integer representing the square resolution height and width, or
    a (height, width) tuple. All of the images in the dataset will be resized to this resolution unless the
    `aspect_ratio_buckets` config is set.
    """

    center_crop: bool = True
    """If True, input images will be center-cropped to the target resolution.
    If False, input images will be randomly cropped to the target resolution.
    """

    random_flip: bool = False
    """Whether random flip augmentations should be applied to input images.
    """

    dataloader_num_workers: int = 0
    """Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    """


class TextualInversionSDDataLoaderConfig(ConfigBaseModel):
    type: Literal["TEXTUAL_INVERSION_SD_DATA_LOADER"] = "TEXTUAL_INVERSION_SD_DATA_LOADER"

    dataset: ImageDirDatasetConfig | ImageCaptionDatasetConfig

    captions: TextualInversionCaptionConfig
    """The caption configuration. One of:

    - [`TextualInversionPresetCaptionTransformConfig`][invoke_training.config.shared.data.transform_config.TextualInversionPresetCaptionTransformConfig]: Use preset `object` or `style` caption templates.
    - [`TextualInversionCaptionTransformConfig`][invoke_training.config.shared.data.transform_config.TextualInversionCaptionTransformConfig]: Use custom caption templates.
    - [`TextualInversionCaptionPrefixTransformConfig`][invoke_training.config.shared.data.transform_config.TextualInversionCaptionPrefixTransformConfig]: Prepend the textual inversion token(s) to all existing dataset captions.
    """  # noqa: E501

    aspect_ratio_buckets: AspectRatioBucketConfig | None = None
    """The aspect ratio bucketing configuration. If None, aspect ratio bucketing is disabled, and all images will be
    resized to the same resolution.
    """

    resolution: int | tuple[int, int] = 512
    """The resolution for input images. Either a scalar integer representing the square resolution height and width, or
    a (height, width) tuple. All of the images in the dataset will be resized to this resolution unless the
    `aspect_ratio_buckets` config is set.
    """

    center_crop: bool = True
    """If True, input images will be center-cropped to the target resolution.
    If False, input images will be randomly cropped to the target resolution.
    """

    random_flip: bool = False
    """Whether random flip augmentations should be applied to input images.
    """

    shuffle_caption_transform: Optional[ShuffleCaptionTransformConfig] = None
    """The caption shuffling configuration. If None, caption shuffling is disabled.
    """

    dataloader_num_workers: int = 0
    """Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    """

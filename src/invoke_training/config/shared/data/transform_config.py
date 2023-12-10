from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class SDImageTransformConfig(BaseModel):
    # The resolution for input images. All of the images in the dataset will be resized to this (square) resolution.
    resolution: int = 512

    # If True, input images will be center-cropped to resolution.
    # If False, input images will be randomly cropped to resolution.
    center_crop: bool = True

    # Whether random flip augmentations should be applied to input images.
    random_flip: bool = False


class SDXLImageTransformConfig(BaseModel):
    # The resolution for input images. All of the images in the dataset will be resized to this (square) resolution.
    resolution: int = 1024

    # If True, input images will be center-cropped to resolution.
    # If False, input images will be randomly cropped to resolution.
    center_crop: bool = True

    # Whether random flip augmentations should be applied to input images.
    random_flip: bool = False


class TextualInversionCaptionTransformConfig(BaseModel):
    type: Literal["TEXTUAL_INVERSION_CAPTION_TRANSFORM"] = "TEXTUAL_INVERSION_CAPTION_TRANSFORM"

    # A list of caption templates with a single template argument 'slot' in each.
    # E.g.:
    # - "a photo of a {}",
    # - "a rendering of a {}",
    # - "a cropped photo of the {}",
    templates: list[str]


class TextualInversionPresetCaptionTransformConfig(BaseModel):
    type: Literal["TEXTUAL_INVERSION_PRESET_CAPTION_TRANSFORM"] = "TEXTUAL_INVERSION_PRESET_CAPTION_TRANSFORM"

    preset: Literal["style", "object"]


TextualInversionCaptionConfig = Annotated[
    Union[
        TextualInversionCaptionTransformConfig,
        TextualInversionPresetCaptionTransformConfig,
    ],
    Field(discriminator="type"),
]


class ShuffleCaptionTransformConfig(BaseModel):
    # The delimiter to use for caption splitting.
    delimiter: str = ","

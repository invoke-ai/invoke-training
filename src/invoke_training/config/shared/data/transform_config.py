from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class SDImageTransformConfig(BaseModel):
    resolution: int = 512
    """The resolution for input images. All of the images in the dataset will be resized to this (square) resolution.
    """

    center_crop: bool = True
    """If True, input images will be center-cropped to resolution.
    If False, input images will be randomly cropped to resolution.
    """

    random_flip: bool = False
    """Whether random flip augmentations should be applied to input images.
    """


class TextualInversionCaptionTransformConfig(BaseModel):
    type: Literal["TEXTUAL_INVERSION_CAPTION_TRANSFORM"] = "TEXTUAL_INVERSION_CAPTION_TRANSFORM"

    templates: list[str]
    """A list of caption templates with a single template argument 'slot' in each.
    E.g.:
    - "a photo of a {}"
    - "a rendering of a {}"
    - "a cropped photo of the {}"
    """


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
    delimiter: str = ","
    """The delimiter to use for caption splitting."""

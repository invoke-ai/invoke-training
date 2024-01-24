from typing import Annotated, Literal, Union

from pydantic import Field

from invoke_training.config.shared.config_base_model import ConfigBaseModel


class TextualInversionCaptionTransformConfig(ConfigBaseModel):
    type: Literal["TEXTUAL_INVERSION_CAPTION_TRANSFORM"] = "TEXTUAL_INVERSION_CAPTION_TRANSFORM"

    templates: list[str]
    """A list of caption templates with a single template argument 'slot' in each.
    E.g.:

    - "a photo of a {}"
    - "a rendering of a {}"
    - "a cropped photo of the {}"
    """


class TextualInversionPresetCaptionTransformConfig(ConfigBaseModel):
    type: Literal["TEXTUAL_INVERSION_PRESET_CAPTION_TRANSFORM"] = "TEXTUAL_INVERSION_PRESET_CAPTION_TRANSFORM"

    preset: Literal["style", "object"]


class TextualInversionCaptionPrefixTransformConfig(ConfigBaseModel):
    type: Literal["TEXTUAL_INVERSION_CAPTION_PREFIX_TRANSFORM"] = "TEXTUAL_INVERSION_CAPTION_PREFIX_TRANSFORM"


TextualInversionCaptionConfig = Annotated[
    Union[
        TextualInversionCaptionTransformConfig,
        TextualInversionPresetCaptionTransformConfig,
        TextualInversionCaptionPrefixTransformConfig,
    ],
    Field(discriminator="type"),
]


class ShuffleCaptionTransformConfig(ConfigBaseModel):
    delimiter: str = ","
    """The delimiter to use for caption splitting."""

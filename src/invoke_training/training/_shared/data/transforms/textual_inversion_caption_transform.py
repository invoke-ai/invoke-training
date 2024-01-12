import typing

import numpy as np


class TextualInversionCaptionTransform:
    """A simple transform that constructs a caption for each example by combining a caption template with the
    placeholder string.
    """

    def __init__(self, field_name: str, placeholder_str: str, caption_templates: list[str], seed: int = 0):
        self._field_name = field_name
        self._placeholder_str = placeholder_str
        self._caption_templates = caption_templates
        self._rng = np.random.default_rng(seed)

    def __call__(self, data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        caption = self._rng.choice(self._caption_templates).format(self._placeholder_str)
        # Assert that the template was well-formed such that the placeholder string is in the output caption.
        assert self._placeholder_str in caption

        data[self._field_name] = caption
        return data

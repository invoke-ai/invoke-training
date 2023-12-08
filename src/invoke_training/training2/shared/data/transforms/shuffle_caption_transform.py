import typing

import numpy as np


class ShuffleCaptionTransform:
    """A transform that applies shuffle transformations to character-delimited captions.

    Example:
    - Original: "unreal engine, render of sci-fi helmet, dramatic lighting"
    - Shuffled: "render of sci-fi helmet, unreal engine, dramatic lighting"
    """

    def __init__(self, field_name: str, delimiter: str = ",", seed: int = 0):
        self._field_name = field_name
        self._delimiter = delimiter
        self._rng = np.random.default_rng(seed)

    def __call__(self, data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        caption: str = data[self._field_name]
        caption_chunks = caption.split(self._delimiter)
        caption_chunks = [s.strip() for s in caption_chunks]

        self._rng.shuffle(caption_chunks)

        join_str = self._delimiter + " "
        data[self._field_name] = join_str.join(caption_chunks)
        return data

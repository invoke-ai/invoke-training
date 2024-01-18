import typing


class CaptionPrefixTransform:
    """A transform that adds a prefix to all example captions."""

    def __init__(self, caption_field_name: str, prefix: str):
        self._caption_field_name = caption_field_name
        self._prefix = prefix

    def __call__(self, data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        data[self._caption_field_name] = self._prefix + data[self._caption_field_name]
        return data

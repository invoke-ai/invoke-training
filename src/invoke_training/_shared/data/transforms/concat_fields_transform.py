import typing


class ConcatFieldsTransform:
    """A transform that concatenate multiple string fields."""

    def __init__(self, src_field_names: list[str], dst_field_name: str, separator: str = " "):
        self._src_field_names = src_field_names
        self._dst_field_name = dst_field_name
        self._separator = separator

    def __call__(self, data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        result = self._separator.join([data[field_name] for field_name in self._src_field_names])
        data[self._dst_field_name] = result
        return data

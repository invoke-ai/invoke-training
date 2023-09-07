import typing


class ConstantFieldTransform:
    """A simple transform that adds a constant field to every example."""

    def __init__(self, field_name: str, field_value: typing.Any):
        self._field_name = field_name
        self._field_value = field_value

    def __call__(self, data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        data[self._field_name] = self._field_value
        return data

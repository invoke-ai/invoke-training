import typing


class DropFieldTransform:
    """A simple transform that drops a field from an example."""

    def __init__(self, field_to_drop: str):
        self._field_to_drop = field_to_drop

    def __call__(self, data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        if self._field_to_drop in data:
            del data[self._field_to_drop]
        return data

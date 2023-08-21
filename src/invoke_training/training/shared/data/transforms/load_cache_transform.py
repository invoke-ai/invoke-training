import typing

from invoke_training.training.shared.data.transforms.tensor_disk_cache import (
    TensorDiskCache,
)


class LoadCacheTransform:
    """A transform that loads data from a TensorDiskCache."""

    def __init__(self, cache: TensorDiskCache, cache_key_field: str, output_field: str):
        self._cache = cache
        self._cache_key_field = cache_key_field
        self._output_field = output_field

    def __call__(self, data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        key = data[self._cache_key_field]
        data[self._output_field] = self._cache.load(hash(key))
        return data

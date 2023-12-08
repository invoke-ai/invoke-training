import typing

from invoke_training.training2.shared.data.transforms.tensor_disk_cache import TensorDiskCache


class LoadCacheTransform:
    """A transform that loads data from a TensorDiskCache."""

    def __init__(
        self, cache: TensorDiskCache, cache_key_field: str, cache_field_to_output_field: typing.Dict[str, str]
    ):
        """Initialize LoadCacheTransform.

        Args:
            cache (TensorDiskCache): The cache to load from.
            cache_key_field (str): The name of the field to use as the cache key.
            cache_field_to_output_field (typing.Dict[str, str]): A map of field names in the cached data to the field
                names where they should be inserted in the example data.
        """
        self._cache = cache
        self._cache_key_field = cache_key_field
        self._cache_field_to_output_field = cache_field_to_output_field

    def __call__(self, data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        key = data[self._cache_key_field]

        cache_data = self._cache.load(key)

        for src, dst in self._cache_field_to_output_field.items():
            data[dst] = cache_data[src]

        return data

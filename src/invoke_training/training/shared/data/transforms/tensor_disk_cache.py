import os
import typing

import torch


class TensorDiskCache:
    """A data cache that caches `torch.Tensor`s on disk."""

    def __init__(self, cache_dir: str):
        super().__init__()
        self._cache_dir = cache_dir

        os.makedirs(self._cache_dir, exist_ok=True)

    def _get_path(self, key: int):
        """Get the cache file path for `key`.
        Args:
            key (int): The cache key.
        Returns:
            str: The cache file path.
        """
        return os.path.join(self._cache_dir, f"{key}.pt")

    def save(self, key: int, data: typing.Dict[str, torch.Tensor]):
        """Save data in the cache.
        Raises:
            AssertionError: If an entry already exists in the cache for this `key`.
        Args:
            key (int): The cache key.
            data (typing.Dict[str, torch.Tensor]): The data to save.
        """
        save_path = self._get_path(key)
        assert not os.path.exists(save_path)
        torch.save(data, save_path)

    def load(self, key: int) -> typing.Dict[str, torch.Tensor]:
        """Load data from the cache.
        Args:
            key (int): The cache key to load.
        Returns:
            typing.Dict[str, torch.Tensor]: Data loaded from the cache.
        """
        return torch.load(self._get_path(key))

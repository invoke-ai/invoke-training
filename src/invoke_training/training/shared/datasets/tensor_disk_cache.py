import os
import typing

import torch

from invoke_training.training.shared.datasets.base_data_cache import BaseDataCache


class TensorDiskCache(BaseDataCache):
    """A data cache that caches `torch.Tensor`s on disk."""

    def __init__(self, cache_dir: str):
        super().__init__()
        self._cache_dir = cache_dir

        os.makedirs(self._cache_dir, exist_ok=True)

    def _get_path(self, idx: int):
        """Get the cache file path for `idx`.

        Args:
            idx (int): The cache index.

        Returns:
            str: The cache file path.
        """
        return os.path.join(self._cache_dir, f"{idx}.pt")

    def save(self, idx: int, data: typing.Dict[str, torch.Tensor]):
        """Save data in the cache.

        Raises:
            AssertionError: If an entry already exists in the cache for this `idx`.

        Args:
            idx (int): The data index (i.e. the cache key).
            data (typing.Dict[str, torch.Tensor]): The data to save.
        """
        save_path = self._get_path(idx)
        assert not os.path.exists(save_path)
        torch.save(data, save_path)

    def load(self, idx: int) -> typing.Dict[str, torch.Tensor]:
        """Load data from the cache.

        Args:
            idx (int): The data index to load (i.e. the cache key).

        Returns:
            typing.Dict[str, torch.Tensor]: Data loaded from the cache.
        """
        return torch.load(self._get_path(idx))

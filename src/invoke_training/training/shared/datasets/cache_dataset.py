from invoke_training.training.shared.datasets.base_data_cache import BaseDataCache


class CacheDataset:
    """A dataset that loads examples from a BaseDataCache."""

    def __init__(self, cache: BaseDataCache):
        self._cache = cache

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int):
        return self._cache.load(idx)

import typing


class BaseDataCache:
    """A base interface that all data caches should implement.

    The intention is that sub-classes will implement cacheing for various data types and cache locations (e.g. memory,
    disk, etc.)
    """

    def __len__(self) -> int:
        """Get the number of entries in the cache.

        Returns:
            int: The length of the cache.
        """
        raise NotImplementedError("__len__ is not implemented.")

    def save(self, idx: int, data: typing.Dict[str, typing.Any]):
        """Save data in the cache.

        Args:
            idx (int): The data index (i.e. the cache key).
            data (typing.Dict[str, typing.Any]): The data to save.
        """
        raise NotImplementedError("save(...) is not implemented.")

    def load(self, idx: int) -> typing.Dict[str, typing.Any]:
        """Load data from the cache.

        Args:
            idx (int): The data index to load (i.e. the cache key).

        Returns:
            typing.Dict[str, typing.Any]: Data loaded from the cache.
        """
        raise NotImplementedError("load(...) is not implemented.")

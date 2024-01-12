import typing

from torch.utils.data import Sampler


class OffsetSampler(Sampler[int]):
    """A sampler that wraps another sampler and applies an offset to all returned values."""

    def __init__(self, sampler: Sampler[int], offset: int):
        self._sampler = sampler
        self._offset = offset

    def __iter__(self) -> typing.Iterator[int]:
        for idx in self._sampler:
            yield idx + self._offset

    def __len__(self) -> int:
        return len(self._sampler)

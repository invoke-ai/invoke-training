import typing

from torch.utils.data import Sampler


class BatchOffsetSampler(Sampler[int]):
    """A sampler that wraps a batch sampler and applies an offset to all returned batch elements."""

    def __init__(self, sampler: Sampler[int], offset: int):
        self._sampler = sampler
        self._offset = offset

    def __iter__(self) -> typing.Iterator[int]:
        for batch in self._sampler:
            offset_batch = [x + self._offset for x in batch]
            yield offset_batch

    def __len__(self) -> int:
        return len(self._sampler)

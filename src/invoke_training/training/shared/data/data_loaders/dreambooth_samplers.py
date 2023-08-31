import typing

import torch
from torch.utils.data import Sampler


class SequentialRangeSampler(Sampler[int]):
    """A sampler that returns sequential values from a range."""

    def __init__(self, start: int, end: int):
        self._start = start
        self._end = end

    def __iter__(self) -> typing.Iterator[int]:
        return iter(range(self._start, self._end))

    def __len__(self) -> int:
        return self._end - self._start


class ShuffledRangeSampler(Sampler[int]):
    """A sampler that returns shuffled values from a range."""

    def __init__(self, range_start: int, range_end: int, generator: torch.Generator = None):
        self._range_start = range_start
        self._range_end = range_end
        self._generator = generator

    def __iter__(self) -> typing.Iterator[int]:
        if self._generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self._generator

        indexes = torch.randperm(self._range_end - self._range_start, generator=generator) + self._range_start
        yield from indexes.tolist()

    def __len__(self) -> int:
        return self._range_end - self._range_start


class InterleavedSampler(Sampler[int]):
    """A meta-Sampler that interleaves multiple samplers.

    The length of this sampler is based on the length of the shortest input sampler. All samplers will contribute the
    same number of samples to the interleaved output.

    Example:
        sampler 1:           ABCD
        sampler 2:           EFG
        sampler 3:           HIJKLM
        interleaved sampler: AEHBFICGJ
    """

    def __init__(self, samplers: list[Sampler[int]]) -> None:
        self._samplers = samplers
        self._min_sampler_len = min([len(s) for s in self._samplers])

    def __iter__(self) -> typing.Iterator[list[int]]:
        sampler_iters = [iter(s) for s in self._samplers]
        while True:
            samples = []
            for sampler_iter in sampler_iters:
                try:
                    samples.append(next(sampler_iter))
                except StopIteration:
                    # The end of the shortest sampler has been reached.
                    return

            yield from samples

    def __len__(self) -> int:
        return self._min_sampler_len * len(self._samplers)

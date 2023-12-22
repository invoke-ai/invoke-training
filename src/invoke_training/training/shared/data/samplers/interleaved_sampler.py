import typing

from torch.utils.data import Sampler

T_co = typing.TypeVar("T_co", covariant=True)


class InterleavedSampler(Sampler[T_co]):
    """A meta-Sampler that interleaves multiple samplers.

    The length of this sampler is based on the length of the shortest input sampler. All samplers will contribute the
    same number of samples to the interleaved output.

    Example:
        sampler 1:           ABCD
        sampler 2:           EFG
        sampler 3:           HIJKLM
        interleaved sampler: AEHBFICGJ
    """

    def __init__(self, samplers: list[Sampler[T_co] | typing.Iterable[T_co]]) -> None:
        self._samplers = samplers
        self._min_sampler_len = min([len(s) for s in self._samplers])

    def __iter__(self) -> typing.Iterator[T_co]:
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

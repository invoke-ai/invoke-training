import itertools
import typing

from torch.utils.data import Sampler

T_co = typing.TypeVar("T_co", covariant=True)


class ConcatSampler(Sampler[T_co]):
    """A meta-Sampler that concatenates multiple samplers.

    Example:
        sampler 1:           ABCD
        sampler 2:           EFG
        sampler 3:           HIJKLM
        ConcatSampler:       ABCDEFGHIJKLM
    """

    def __init__(self, samplers: list[Sampler[T_co] | typing.Iterable[T_co]]) -> None:
        self._samplers = samplers

    def __iter__(self) -> typing.Iterator[T_co]:
        return itertools.chain(*self._samplers)

    def __len__(self) -> int:
        return sum([len(s) for s in self._samplers])

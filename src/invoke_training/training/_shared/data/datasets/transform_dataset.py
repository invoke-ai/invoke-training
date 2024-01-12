import typing

import torch.utils.data

# The data type expected to be produced by the base dataset and handled by transforms.
DataType = typing.Dict[str, typing.Any]

TransformType = typing.Callable[[DataType], DataType]


class TransformDataset(torch.utils.data.Dataset):
    """A Dataset that wraps a base dataset and applies callable transforms to its outputs."""

    def __init__(self, base_dataset: torch.utils.data.Dataset, transforms: list[TransformType]) -> None:
        super().__init__()
        self._base_dataset = base_dataset
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._base_dataset)

    def __getitem__(self, idx: int) -> DataType:
        example = self._base_dataset[idx]
        for t in self._transforms:
            example = t(example)
        return example

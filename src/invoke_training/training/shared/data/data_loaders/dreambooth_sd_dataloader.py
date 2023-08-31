import typing

import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from transformers import CLIPTokenizer

from invoke_training.training.config.data_config import ImageDirDatasetConfig
from invoke_training.training.shared.data.datasets.image_dir_dataset import (
    ImageDirDataset,
)
from invoke_training.training.shared.data.datasets.transform_dataset import (
    TransformDataset,
)
from invoke_training.training.shared.data.transforms.constant_field_transform import (
    ConstantFieldTransform,
)
from invoke_training.training.shared.data.transforms.sd_image_transform import (
    SDImageTransform,
)
from invoke_training.training.shared.data.transforms.sd_tokenize_transform import (
    SDTokenizeTransform,
)


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


def build_dreambooth_sd_dataloader(
    instance_prompt: str,
    instance_dataset_config: ImageDirDatasetConfig,
    class_prompt: typing.Optional[str],
    class_data_dir: typing.Optional[str],
    tokenizer: typing.Optional[CLIPTokenizer],
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Construct a DataLoader for a DreamBooth dataset for Stable Diffusion v1/v2.."""

    # 1. Prepare instance dataset
    instance_dataset = ImageDirDataset(instance_dataset_config.dataset_dir)
    instance_dataset = TransformDataset(
        instance_dataset,
        [ConstantFieldTransform("caption", instance_prompt), ConstantFieldTransform("loss_weight", 1.0)],
    )

    # 2. Prepare class dataset.
    class_dataset = None
    if class_data_dir is not None:
        class_dataset = ImageDirDataset(class_data_dir)
        class_dataset = TransformDataset(
            class_dataset,
            [ConstantFieldTransform("caption", class_prompt), ConstantFieldTransform("loss_weight", 1.0)],
        )

    # 3. Merge instance dataset and class dataset.
    merged_dataset = ConcatDataset([instance_dataset, class_dataset])
    all_transforms = [
        SDImageTransform(
            resolution=instance_dataset_config.image_transforms.resolution,
            center_crop=instance_dataset_config.image_transforms.center_crop,
            random_flip=instance_dataset_config.image_transforms.random_flip,
        ),
        SDTokenizeTransform(tokenizer),
    ]
    merged_dataset = TransformDataset(merged_dataset, all_transforms)

    # 4. Prepare instance dataset sampler. Note that the instance_dataset comes first in the merged_dataset.
    samplers = []
    if shuffle:
        samplers.append(SequentialRangeSampler(0, len(instance_dataset)))
    else:
        samplers.append(ShuffledRangeSampler(0, len(instance_dataset)))

    # 5. Prepare class dataset sampler. Note that the class_dataset comes first in the merged_dataset.
    if class_dataset is not None:
        if shuffle:
            samplers.append(SequentialRangeSampler(len(instance_dataset), len(instance_dataset) + len(class_dataset)))
        else:
            samplers.append(ShuffledRangeSampler(len(instance_dataset), len(instance_dataset) + len(class_dataset)))

    # 6. Interleave instance and class samplers.
    interleaved_sampler = InterleavedSampler(samplers)

    return DataLoader(
        merged_dataset,
        sampler=interleaved_sampler,
        batch_size=batch_size,
        num_workers=instance_dataset_config.dataloader_num_workers,
    )

import typing

from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from invoke_training.training.config.data_config import ImageDirDatasetConfig
from invoke_training.training.shared.data.datasets.dreambooth_dataset import (
    DreamBoothDataset,
)
from invoke_training.training.shared.data.datasets.image_dir_dataset import (
    ImageDirDataset,
)
from invoke_training.training.shared.data.datasets.transform_dataset import (
    TransformDataset,
)
from invoke_training.training.shared.data.transforms.sd_image_transform import (
    SDImageTransform,
)
from invoke_training.training.shared.data.transforms.sd_tokenize_transform import (
    SDTokenizeTransform,
)


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

    instance_dataset = ImageDirDataset(instance_dataset_config.dataset_dir)

    class_dataset = None
    if class_data_dir is not None:
        class_dataset = ImageDirDataset(class_data_dir)

    dataset = DreamBoothDataset(
        instance_dataset=instance_dataset,
        instance_prompt=instance_prompt,
        class_dataset=class_dataset,
        class_prompt=class_prompt,
        balance_datasets=True,
        shuffle=True,
    )

    all_transforms = []
    all_transforms.append(
        SDImageTransform(
            resolution=instance_dataset_config.image_transforms.resolution,
            center_crop=instance_dataset_config.image_transforms.center_crop,
            random_flip=instance_dataset_config.image_transforms.random_flip,
        )
    )

    all_transforms.append(SDTokenizeTransform(tokenizer))

    dataset = TransformDataset(dataset, all_transforms)

    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=instance_dataset_config.dataloader_num_workers,
    )

import typing

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from invoke_training.training.config.data_config import ImageDirDatasetConfig
from invoke_training.training.shared.data.data_loaders.image_caption_sdxl_dataloader import (
    sdxl_image_caption_collate_fn,
)
from invoke_training.training.shared.data.datasets.dreambooth_dataset import (
    DreamBoothDataset,
)
from invoke_training.training.shared.data.datasets.image_dir_dataset import (
    ImageDirDataset,
)
from invoke_training.training.shared.data.datasets.transform_dataset import (
    TransformDataset,
)
from invoke_training.training.shared.data.transforms.sd_tokenize_transform import (
    SDTokenizeTransform,
)
from invoke_training.training.shared.data.transforms.sdxl_image_transform import (
    SDXLImageTransform,
)


def build_dreambooth_sdxl_dataloader(
    instance_prompt: str,
    instance_dataset_config: ImageDirDatasetConfig,
    class_prompt: typing.Optional[str],
    class_data_dir: typing.Optional[str],
    tokenizer_1: PreTrainedTokenizer,
    tokenizer_2: PreTrainedTokenizer,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Construct a DataLoader for a DreamBooth dataset for Stable Diffusion XL."""

    instance_dataset = ImageDirDataset(instance_dataset_config.dataset_dir)

    class_dataset = None
    if class_data_dir is not None:
        class_dataset = ImageDirDataset(class_data_dir)

    dataset = DreamBoothDataset(
        instance_dataset=instance_dataset,
        instance_prompt=instance_prompt,
        class_dataset=class_dataset,
        class_prompt=class_prompt,
        shuffle=True,
    )

    all_transforms = []
    all_transforms.append(
        SDXLImageTransform(
            resolution=instance_dataset_config.image_transforms.resolution,
            center_crop=instance_dataset_config.image_transforms.center_crop,
            random_flip=instance_dataset_config.image_transforms.random_flip,
        )
    )
    all_transforms.append(
        SDTokenizeTransform(tokenizer_1, src_caption_key="caption", dst_token_key="caption_token_ids_1")
    )
    all_transforms.append(
        SDTokenizeTransform(tokenizer_2, src_caption_key="caption", dst_token_key="caption_token_ids_2")
    )

    dataset = TransformDataset(dataset, all_transforms)

    return DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=sdxl_image_caption_collate_fn,
        batch_size=batch_size,
        num_workers=instance_dataset_config.dataloader_num_workers,
    )

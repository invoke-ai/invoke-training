from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from invoke_training.training.finetune_lora.finetune_lora_config import DatasetConfig
from invoke_training.training.shared.data.datasets.hf_dir_image_caption_dataset import (
    HFDirImageCaptionDataset,
)
from invoke_training.training.shared.data.datasets.hf_hub_image_caption_dataset import (
    HFHubImageCaptionDataset,
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


def build_image_caption_sd_dataloader(config: DatasetConfig, tokenizer: CLIPTokenizer, batch_size: int) -> DataLoader:
    """Construct a DataLoader for an image-caption dataset for Stable Diffusion v1/v2..

    Args:
        config (DatasetConfig): The dataset config.
        tokenizer (CLIPTokenizer): The tokenizer to apply to the captions.
        batch_size (int): The DataLoader batch size.

    Returns:
        DataLoader
    """

    if config.dataset_name is not None:
        base_dataset = HFHubImageCaptionDataset(
            dataset_name=config.dataset_name,
            hf_load_dataset_kwargs={
                "name": config.dataset_config_name,
                "cache_dir": config.hf_cache_dir,
            },
            image_column=config.image_column,
            caption_column=config.caption_column,
        )
    elif config.dataset_dir is not None:
        base_dataset = HFDirImageCaptionDataset(
            dataset_dir=config.dataset_dir,
            hf_load_dataset_kwargs=None,
            image_column=config.image_column,
            caption_column=config.caption_column,
        )
    else:
        raise ValueError("One of 'dataset_name' or 'dataset_dir' must be set.")

    image_transform = SDImageTransform(
        resolution=config.resolution, center_crop=config.center_crop, random_flip=config.random_flip
    )
    tokenize_transform = SDTokenizeTransform(tokenizer)

    dataset = TransformDataset(base_dataset, [image_transform, tokenize_transform])

    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=config.dataloader_num_workers,
    )

import typing

import torch
from torch.utils.data import DataLoader

from invoke_training.config.shared.data.data_loader_config import ImageCaptionSDXLDataLoaderConfig
from invoke_training.config.shared.data.dataset_config import (
    HFDirImageCaptionDatasetConfig,
    HFHubImageCaptionDatasetConfig,
)
from invoke_training.training.shared.data.datasets.build_dataset import (
    build_hf_dir_image_caption_dataset,
    build_hf_hub_image_caption_dataset,
)
from invoke_training.training.shared.data.datasets.transform_dataset import TransformDataset
from invoke_training.training.shared.data.transforms.drop_field_transform import DropFieldTransform
from invoke_training.training.shared.data.transforms.load_cache_transform import LoadCacheTransform
from invoke_training.training.shared.data.transforms.sdxl_image_transform import SDXLImageTransform
from invoke_training.training.shared.data.transforms.tensor_disk_cache import TensorDiskCache


def sdxl_image_caption_collate_fn(examples):
    """A batch collation function for the image-caption SDXL data loader."""
    out_examples = {
        "id": [example["id"] for example in examples],
    }

    if "image" in examples[0]:
        out_examples["image"] = torch.stack([example["image"] for example in examples])

    if "original_size_hw" in examples[0]:
        out_examples["original_size_hw"] = [example["original_size_hw"] for example in examples]

    if "crop_top_left_yx" in examples[0]:
        out_examples["crop_top_left_yx"] = [example["crop_top_left_yx"] for example in examples]

    if "caption" in examples[0]:
        out_examples["caption"] = [example["caption"] for example in examples]

    if "loss_weight" in examples[0]:
        out_examples["loss_weight"] = torch.tensor([example["loss_weight"] for example in examples])

    if "prompt_embeds" in examples[0]:
        out_examples["prompt_embeds"] = torch.stack([example["prompt_embeds"] for example in examples])
        out_examples["pooled_prompt_embeds"] = torch.stack([example["pooled_prompt_embeds"] for example in examples])

    if "vae_output" in examples[0]:
        out_examples["vae_output"] = torch.stack([example["vae_output"] for example in examples])

    return out_examples


def build_image_caption_sdxl_dataloader(
    config: ImageCaptionSDXLDataLoaderConfig,
    batch_size: int,
    text_encoder_output_cache_dir: typing.Optional[str] = None,
    vae_output_cache_dir: typing.Optional[str] = None,
    shuffle: bool = True,
) -> DataLoader:
    """Construct a DataLoader for an image-caption dataset for Stable Diffusion XL.

    Args:
        config (ImageCaptionSDXLDataLoaderConfig): The dataset config.
        tokenizer (CLIPTokenizer): The tokenizer to apply to the captions. Can be None if
            `text_encoder_output_cache_dir` is set.
        batch_size (int): The DataLoader batch size.
        text_encoder_output_cache_dir (str, optional): The directory where text encoder outputs are cached and should be
            loaded from. If set, then the TokenizeTransform will not be applied.
        vae_output_cache_dir (str, optional): The directory where VAE outputs are cached and should be loaded from. If
            set, then the image augmentation transforms will be skipped, and the image will not be copied to VRAM.
        shuffle (bool, optional): Whether to shuffle the dataset order.
    Returns:
        DataLoader
    """
    if isinstance(config.dataset, HFHubImageCaptionDatasetConfig):
        base_dataset = build_hf_hub_image_caption_dataset(config.dataset)
    elif isinstance(config.dataset, HFDirImageCaptionDatasetConfig):
        base_dataset = build_hf_dir_image_caption_dataset(config)
    else:
        raise ValueError(f"Unexpected dataset config type: '{type(config.dataset)}'.")

    all_transforms = []
    if vae_output_cache_dir is None:
        all_transforms.append(
            SDXLImageTransform(
                resolution=config.image_transforms.resolution,
                center_crop=config.image_transforms.center_crop,
                random_flip=config.image_transforms.random_flip,
            )
        )
    else:
        vae_cache = TensorDiskCache(vae_output_cache_dir)
        all_transforms.append(
            LoadCacheTransform(
                cache=vae_cache,
                cache_key_field="id",
                cache_field_to_output_field={
                    "vae_output": "vae_output",
                    "original_size_hw": "original_size_hw",
                    "crop_top_left_yx": "crop_top_left_yx",
                },
            )
        )
        # We drop the image to avoid having to either convert from PIL, or handle PIL batch collation.
        all_transforms.append(DropFieldTransform("image"))

    if text_encoder_output_cache_dir is not None:
        text_encoder_cache = TensorDiskCache(text_encoder_output_cache_dir)
        all_transforms.append(
            LoadCacheTransform(
                cache=text_encoder_cache,
                cache_key_field="id",
                cache_field_to_output_field={
                    "prompt_embeds": "prompt_embeds",
                    "pooled_prompt_embeds": "pooled_prompt_embeds",
                },
            )
        )

    dataset = TransformDataset(base_dataset, all_transforms)

    return DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=sdxl_image_caption_collate_fn,
        batch_size=batch_size,
        num_workers=config.dataloader_num_workers,
    )

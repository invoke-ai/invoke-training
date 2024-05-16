import typing

import torch
from torch.utils.data import DataLoader

from invoke_training._shared.data.datasets.build_dataset import (
    build_hf_hub_image_caption_dataset,
    build_image_caption_dir_dataset,
    build_image_caption_jsonl_dataset,
)
from invoke_training._shared.data.datasets.transform_dataset import TransformDataset
from invoke_training._shared.data.samplers.aspect_ratio_bucket_batch_sampler import AspectRatioBucketBatchSampler
from invoke_training._shared.data.transforms.caption_prefix_transform import CaptionPrefixTransform
from invoke_training._shared.data.transforms.drop_field_transform import DropFieldTransform
from invoke_training._shared.data.transforms.load_cache_transform import LoadCacheTransform
from invoke_training._shared.data.transforms.sd_image_transform import SDImageTransform
from invoke_training._shared.data.transforms.tensor_disk_cache import TensorDiskCache
from invoke_training._shared.data.utils.aspect_ratio_bucket_manager import AspectRatioBucketManager
from invoke_training.config.data.data_loader_config import AspectRatioBucketConfig, ImageCaptionSDDataLoaderConfig
from invoke_training.config.data.dataset_config import (
    HFHubImageCaptionDatasetConfig,
    ImageCaptionDirDatasetConfig,
    ImageCaptionJsonlDatasetConfig,
)


def sd_image_caption_collate_fn(examples):
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

    if "text_encoder_output" in examples[0]:
        out_examples["text_encoder_output"] = torch.stack([example["text_encoder_output"] for example in examples])

    if "vae_output" in examples[0]:
        out_examples["vae_output"] = torch.stack([example["vae_output"] for example in examples])

    if "mask" in examples[0]:
        out_examples["mask"] = torch.stack([example["mask"] for example in examples])

    return out_examples


def build_aspect_ratio_bucket_manager(config: AspectRatioBucketConfig):
    return AspectRatioBucketManager.from_constraints(
        target_resolution=config.target_resolution,
        start_dim=config.start_dim,
        end_dim=config.end_dim,
        divisible_by=config.divisible_by,
    )


def build_image_caption_sd_dataloader(
    config: ImageCaptionSDDataLoaderConfig,
    batch_size: int,
    text_encoder_output_cache_dir: typing.Optional[str] = None,
    text_encoder_cache_field_to_output_field: typing.Optional[dict[str, str]] = None,
    vae_output_cache_dir: typing.Optional[str] = None,
    shuffle: bool = True,
) -> DataLoader:
    """Construct a DataLoader for an image-caption dataset for Stable Diffusion XL.

    Args:
        config (ImageCaptionSDDataLoaderConfig): The dataset config.
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
    elif isinstance(config.dataset, ImageCaptionJsonlDatasetConfig):
        base_dataset = build_image_caption_jsonl_dataset(config.dataset)
    elif isinstance(config.dataset, ImageCaptionDirDatasetConfig):
        base_dataset = build_image_caption_dir_dataset(config.dataset)
    else:
        raise ValueError(f"Unexpected dataset config type: '{type(config.dataset)}'.")

    # Initialize either the fixed target resolution or aspect ratio buckets.
    if config.aspect_ratio_buckets is None:
        target_resolution = config.resolution
        aspect_ratio_bucket_manager = None
        batch_sampler = None
    else:
        target_resolution = None
        aspect_ratio_bucket_manager = build_aspect_ratio_bucket_manager(config=config.aspect_ratio_buckets)
        # TODO(ryand): Drill-down the seed parameter rather than hard-coding to 0 here.
        batch_sampler = AspectRatioBucketBatchSampler.from_image_sizes(
            bucket_manager=aspect_ratio_bucket_manager,
            image_sizes=base_dataset.get_image_dimensions(),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=0,
        )

    all_transforms = []

    if config.caption_prefix is not None:
        all_transforms.append(CaptionPrefixTransform(caption_field_name="caption", prefix=config.caption_prefix + " "))

    if vae_output_cache_dir is None:
        all_transforms.append(
            SDImageTransform(
                resolution=target_resolution,
                aspect_ratio_bucket_manager=aspect_ratio_bucket_manager,
                center_crop=config.center_crop,
                random_flip=config.random_flip,
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
        assert text_encoder_cache_field_to_output_field is not None
        text_encoder_cache = TensorDiskCache(text_encoder_output_cache_dir)
        all_transforms.append(
            LoadCacheTransform(
                cache=text_encoder_cache,
                cache_key_field="id",
                cache_field_to_output_field=text_encoder_cache_field_to_output_field,
            )
        )

    dataset = TransformDataset(base_dataset, all_transforms)

    if batch_sampler is None:
        return DataLoader(
            dataset,
            shuffle=shuffle,
            collate_fn=sd_image_caption_collate_fn,
            batch_size=batch_size,
            num_workers=config.dataloader_num_workers,
        )
    else:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=sd_image_caption_collate_fn,
            num_workers=config.dataloader_num_workers,
        )

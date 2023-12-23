import typing

import torch
from torch.utils.data import DataLoader

from invoke_training.config.shared.data.data_loader_config import ImagePairPreferenceSDDataLoaderConfig
from invoke_training.training.shared.data.datasets.build_dataset import (
    build_hf_image_pair_preference_dataset,
)
from invoke_training.training.shared.data.datasets.transform_dataset import TransformDataset
from invoke_training.training.shared.data.transforms.load_cache_transform import LoadCacheTransform
from invoke_training.training.shared.data.transforms.sd_image_transform import SDImageTransform
from invoke_training.training.shared.data.transforms.tensor_disk_cache import TensorDiskCache


def sd_image_pair_preference_collate_fn(examples):
    """A batch collation function."""

    stack_keys = {"image_0", "image_1", "prompt_embeds", "pooled_prompt_embeds", "text_encoder_output", "vae_output"}
    list_keys = {
        "id",
        "original_size_hw_0",
        "original_size_hw_1",
        "crop_top_left_yx_0",
        "crop_top_left_yx_1",
        "prefer_0",
        "prefer_1",
        "caption",
    }

    unhandled_keys = set(examples[0].keys()) - (stack_keys | list_keys)
    if len(unhandled_keys) > 0:
        raise ValueError(f"The following keys are not handled by the collate function: {unhandled_keys}.")

    out_examples = {}

    # torch.stack(...)
    for k in stack_keys:
        if k in examples[0]:
            out_examples[k] = torch.stack([example[k] for example in examples])

    # Basic list.
    for k in list_keys:
        if k in examples[0]:
            out_examples[k] = [example[k] for example in examples]

    return out_examples


def build_image_pair_preference_sd_dataloader(
    config: ImagePairPreferenceSDDataLoaderConfig,
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
    base_dataset = build_hf_image_pair_preference_dataset(config=config.dataset)

    target_resolution = config.image_transforms.resolution

    all_transforms = []
    if vae_output_cache_dir is None:
        all_transforms.append(
            SDImageTransform(
                resolution=target_resolution,
                aspect_ratio_bucket_manager=None,
                center_crop=config.image_transforms.center_crop,
                random_flip=config.image_transforms.random_flip,
                image_field_name="image_0",
                orig_size_field_name="original_size_hw_0",
                crop_field_name="crop_top_left_yx_0",
            )
        )
        all_transforms.append(
            SDImageTransform(
                resolution=target_resolution,
                aspect_ratio_bucket_manager=None,
                center_crop=config.image_transforms.center_crop,
                random_flip=config.image_transforms.random_flip,
                image_field_name="image_1",
                orig_size_field_name="original_size_hw_1",
                crop_field_name="crop_top_left_yx_1",
            )
        )
    else:
        raise NotImplementedError("VAE caching is not yet implemented.")
        # vae_cache = TensorDiskCache(vae_output_cache_dir)
        # all_transforms.append(
        #     LoadCacheTransform(
        #         cache=vae_cache,
        #         cache_key_field="id",
        #         cache_field_to_output_field={
        #             "vae_output": "vae_output",
        #             "original_size_hw": "original_size_hw",
        #             "crop_top_left_yx": "crop_top_left_yx",
        #         },
        #     )
        # )
        # # We drop the image to avoid having to either convert from PIL, or handle PIL batch collation.
        # all_transforms.append(DropFieldTransform("image"))

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

    return DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=sd_image_pair_preference_collate_fn,
        batch_size=batch_size,
        num_workers=config.dataloader_num_workers,
    )

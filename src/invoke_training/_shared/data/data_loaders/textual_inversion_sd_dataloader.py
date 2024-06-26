from typing import Literal, Optional

from torch.utils.data import DataLoader

from invoke_training._shared.data.data_loaders.image_caption_sd_dataloader import (
    build_aspect_ratio_bucket_manager,
    sd_image_caption_collate_fn,
)
from invoke_training._shared.data.datasets.build_dataset import (
    build_hf_hub_image_caption_dataset,
    build_image_caption_dir_dataset,
    build_image_caption_jsonl_dataset,
)
from invoke_training._shared.data.datasets.image_dir_dataset import ImageDirDataset
from invoke_training._shared.data.datasets.transform_dataset import TransformDataset
from invoke_training._shared.data.samplers.aspect_ratio_bucket_batch_sampler import AspectRatioBucketBatchSampler
from invoke_training._shared.data.transforms.concat_fields_transform import ConcatFieldsTransform
from invoke_training._shared.data.transforms.drop_field_transform import DropFieldTransform
from invoke_training._shared.data.transforms.load_cache_transform import LoadCacheTransform
from invoke_training._shared.data.transforms.sd_image_transform import SDImageTransform
from invoke_training._shared.data.transforms.shuffle_caption_transform import ShuffleCaptionTransform
from invoke_training._shared.data.transforms.template_caption_transform import TemplateCaptionTransform
from invoke_training._shared.data.transforms.tensor_disk_cache import TensorDiskCache
from invoke_training.config.data.data_loader_config import TextualInversionSDDataLoaderConfig
from invoke_training.config.data.dataset_config import (
    HFHubImageCaptionDatasetConfig,
    ImageCaptionDirDatasetConfig,
    ImageCaptionJsonlDatasetConfig,
    ImageDirDatasetConfig,
)


def get_preset_ti_caption_templates(preset: Literal["object", "style"]) -> list[str]:
    if preset == "object":
        return [
            "a photo of a {}",
            "a rendering of a {}",
            "a cropped photo of the {}",
            "the photo of a {}",
            "a photo of a clean {}",
            "a photo of a dirty {}",
            "a dark photo of the {}",
            "a photo of my {}",
            "a photo of the cool {}",
            "a close-up photo of a {}",
            "a bright photo of the {}",
            "a cropped photo of a {}",
            "a photo of the {}",
            "a good photo of the {}",
            "a photo of one {}",
            "a close-up photo of the {}",
            "a rendition of the {}",
            "a photo of the clean {}",
            "a rendition of a {}",
            "a photo of a nice {}",
            "a good photo of a {}",
            "a photo of the nice {}",
            "a photo of the small {}",
            "a photo of the weird {}",
            "a photo of the large {}",
            "a photo of a cool {}",
            "a photo of a small {}",
        ]
    elif preset == "style":
        return [
            "a painting in the style of {}",
            "a rendering in the style of {}",
            "a cropped painting in the style of {}",
            "the painting in the style of {}",
            "a clean painting in the style of {}",
            "a dirty painting in the style of {}",
            "a dark painting in the style of {}",
            "a picture in the style of {}",
            "a cool painting in the style of {}",
            "a close-up painting in the style of {}",
            "a bright painting in the style of {}",
            "a good painting in the style of {}",
            "a close-up painting in the style of {}",
            "a rendition in the style of {}",
            "a nice painting in the style of {}",
            "a small painting in the style of {}",
            "a weird painting in the style of {}",
            "a large painting in the style of {}",
            "a photo in the style of {}",
            "an image in the style of {}",
            "a drawing in the style of {}",
            "a sketch in the style of {}",
            "a digital work in the style of {}",
            "a digital rendering in the style of {}",
            "a photograph in the style of {}",
            "photography in the style of {}",
        ]
    else:
        raise ValueError(f"Unrecognized learnable property type: '{preset}'.")


def build_textual_inversion_sd_dataloader(  # noqa: C901
    config: TextualInversionSDDataLoaderConfig,
    placeholder_token: str,
    batch_size: int,
    use_masks: bool = False,
    vae_output_cache_dir: Optional[str] = None,
    shuffle: bool = True,
) -> DataLoader:
    """Construct a DataLoader for a Textual Inversion dataset for Stable Diffusion.

    Args:
        config (TextualInversionSDDataLoaderConfig): The dataset config.
        placeholder_token (str): The placeholder token being trained.
        batch_size (int): The DataLoader batch size.
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
    elif isinstance(config.dataset, ImageDirDatasetConfig):
        base_dataset = ImageDirDataset(
            image_dir=config.dataset.dataset_dir, keep_in_memory=config.dataset.keep_in_memory
        )
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

    if sum([config.caption_templates is not None, config.caption_preset is not None]) != 1:
        raise ValueError("Either caption_templates or caption_preset must be set.")

    if config.caption_templates is not None:
        # Overwrites the caption field. Typically used with a ImageDirDataset that does not have captions.
        caption_tf = TemplateCaptionTransform(
            field_name="caption_prefix" if config.keep_original_captions else "caption",
            placeholder_str=placeholder_token,
            caption_templates=config.caption_templates,
        )
    elif config.caption_preset is not None:
        # Overwrites the caption field. Typically used with a ImageDirDataset that does not have captions.
        caption_tf = TemplateCaptionTransform(
            field_name="caption_prefix" if config.keep_original_captions else "caption",
            placeholder_str=placeholder_token,
            caption_templates=get_preset_ti_caption_templates(config.caption_preset),
        )
    else:
        raise ValueError("Either caption_templates or caption_preset must be set.")

    all_transforms = [caption_tf]

    if config.keep_original_captions:
        # This will only work with a HFHubImageCaptionDataset or HFDirImageCaptionDataset that already has captions.
        all_transforms.append(
            ConcatFieldsTransform(
                src_field_names=["caption_prefix", "caption"], dst_field_name="caption", separator=" "
            )
        )

    if config.shuffle_caption_delimiter is not None:
        all_transforms.append(ShuffleCaptionTransform(field_name="caption", delimiter=config.shuffle_caption_delimiter))

    if vae_output_cache_dir is None:
        image_field_names = ["image"]
        if use_masks:
            image_field_names.append("mask")
        else:
            all_transforms.append(DropFieldTransform("mask"))

        all_transforms.append(
            SDImageTransform(
                image_field_names=image_field_names,
                fields_to_normalize_to_range_minus_one_to_one=["image"],
                resolution=target_resolution,
                aspect_ratio_bucket_manager=aspect_ratio_bucket_manager,
                center_crop=config.center_crop,
                random_flip=config.random_flip,
            )
        )
    else:
        # We drop the image to avoid having to either convert from PIL, or handle PIL batch collation.
        all_transforms.append(DropFieldTransform("image"))
        all_transforms.append(DropFieldTransform("mask"))

        vae_cache = TensorDiskCache(vae_output_cache_dir)

        cache_field_to_output_field = {
            "vae_output": "vae_output",
            "original_size_hw": "original_size_hw",
            "crop_top_left_yx": "crop_top_left_yx",
        }
        if use_masks:
            cache_field_to_output_field["mask"] = "mask"

        all_transforms.append(
            LoadCacheTransform(
                cache=vae_cache,
                cache_key_field="id",
                cache_field_to_output_field=cache_field_to_output_field,
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
            persistent_workers=config.dataloader_num_workers > 0,
        )
    else:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=sd_image_caption_collate_fn,
            num_workers=config.dataloader_num_workers,
            persistent_workers=config.dataloader_num_workers > 0,
        )

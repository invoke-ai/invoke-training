from typing import Optional

from torch.utils.data import DataLoader

from invoke_training.config.shared.data.data_loader_config import TextualInversionSDXLDataLoaderConfig
from invoke_training.config.shared.data.transform_config import (
    TextualInversionCaptionTransformConfig,
    TextualInversionPresetCaptionTransformConfig,
)
from invoke_training.training.shared.data.data_loaders.image_caption_sd_dataloader import (
    sd_image_caption_collate_fn,
)
from invoke_training.training.shared.data.data_loaders.textual_inversion_sd_dataloader import (
    get_preset_ti_caption_templates,
)
from invoke_training.training.shared.data.datasets.image_dir_dataset import ImageDirDataset
from invoke_training.training.shared.data.datasets.transform_dataset import TransformDataset
from invoke_training.training.shared.data.transforms.drop_field_transform import DropFieldTransform
from invoke_training.training.shared.data.transforms.load_cache_transform import LoadCacheTransform
from invoke_training.training.shared.data.transforms.sd_image_transform import SDImageTransform
from invoke_training.training.shared.data.transforms.shuffle_caption_transform import ShuffleCaptionTransform
from invoke_training.training.shared.data.transforms.tensor_disk_cache import TensorDiskCache
from invoke_training.training.shared.data.transforms.textual_inversion_caption_transform import (
    TextualInversionCaptionTransform,
)


def build_textual_inversion_sdxl_dataloader(
    config: TextualInversionSDXLDataLoaderConfig,
    placeholder_str: str,
    batch_size: int,
    vae_output_cache_dir: Optional[str] = None,
    shuffle: bool = True,
) -> DataLoader:
    """Construct a DataLoader for a Textual Inversion dataset for Stable Diffusion v1/v2..

    Args:
        config (TextualInversionSDDataLoaderConfig): The dataset config.
        placeholder_str (str): The placeholder string being trained.
        tokenizer (CLIPTokenizer, option): The tokenizer to apply to the captions. Can be None if
            `text_encoder_output_cache_dir` is set.
        batch_size (int): The DataLoader batch size.
        vae_output_cache_dir (str, optional): The directory where VAE outputs are cached and should be loaded from. If
            set, then the image augmentation transforms will be skipped, and the image will not be copied to VRAM.
        shuffle (bool, optional): Whether to shuffle the dataset order.
    Returns:
        DataLoader
    """

    base_dataset = ImageDirDataset(image_dir=config.dataset.dataset_dir)

    if isinstance(config.captions, TextualInversionCaptionTransformConfig):
        caption_templates = config.captions.templates
    elif isinstance(config.captions, TextualInversionPresetCaptionTransformConfig):
        caption_templates = get_preset_ti_caption_templates(config.captions.preset)
    else:
        raise ValueError(f"Unexpected caption config type: '{type(config.captions)}'.")

    all_transforms = [
        TextualInversionCaptionTransform(
            field_name="caption",
            placeholder_str=placeholder_str,
            caption_templates=caption_templates,
        ),
    ]

    if config.shuffle_caption_transform is not None:
        all_transforms.append(
            ShuffleCaptionTransform(field_name="caption", delimiter=config.shuffle_caption_transform.delimiter)
        )

    if vae_output_cache_dir is None:
        all_transforms.append(
            SDImageTransform(
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

    dataset = TransformDataset(base_dataset, all_transforms)

    return DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=sd_image_caption_collate_fn,
        batch_size=batch_size,
        num_workers=config.dataloader_num_workers,
    )

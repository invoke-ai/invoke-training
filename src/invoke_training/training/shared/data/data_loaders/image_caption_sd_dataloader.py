import typing

from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from invoke_training.training.config.finetune_lora_config import (
    ImageCaptionDataLoaderConfig,
)
from invoke_training.training.shared.data.datasets.hf_dir_image_caption_dataset import (
    HFDirImageCaptionDataset,
)
from invoke_training.training.shared.data.datasets.hf_hub_image_caption_dataset import (
    HFHubImageCaptionDataset,
)
from invoke_training.training.shared.data.datasets.transform_dataset import (
    TransformDataset,
)
from invoke_training.training.shared.data.transforms.drop_field_transform import (
    DropFieldTransform,
)
from invoke_training.training.shared.data.transforms.load_cache_transform import (
    LoadCacheTransform,
)
from invoke_training.training.shared.data.transforms.sd_image_transform import (
    SDImageTransform,
)
from invoke_training.training.shared.data.transforms.sd_tokenize_transform import (
    SDTokenizeTransform,
)
from invoke_training.training.shared.data.transforms.tensor_disk_cache import (
    TensorDiskCache,
)


def build_image_caption_sd_dataloader(
    config: ImageCaptionDataLoaderConfig,
    tokenizer: typing.Optional[CLIPTokenizer],
    batch_size: int,
    text_encoder_output_cache_dir: typing.Optional[str] = None,
    vae_output_cache_dir: typing.Optional[str] = None,
    shuffle: bool = True,
) -> DataLoader:
    """Construct a DataLoader for an image-caption dataset for Stable Diffusion v1/v2..

    Args:
        config (ImageCaptionDataLoaderConfig): The dataset config.
        tokenizer (CLIPTokenizer, option): The tokenizer to apply to the captions. Can be None if
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

    all_transforms = []
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
                cache=vae_cache, cache_key_field="id", cache_field_to_output_field={"vae_output": "vae_output"}
            )
        )
        # We drop the image to avoid having to either convert from PIL, or handle PIL batch collation.
        all_transforms.append(DropFieldTransform("image"))

    if text_encoder_output_cache_dir is None:
        all_transforms.append(SDTokenizeTransform(tokenizer))
    else:
        text_encoder_cache = TensorDiskCache(text_encoder_output_cache_dir)
        all_transforms.append(
            LoadCacheTransform(
                cache=text_encoder_cache,
                cache_key_field="id",
                cache_field_to_output_field={"text_encoder_output": "text_encoder_output"},
            )
        )

    dataset = TransformDataset(base_dataset, all_transforms)

    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=config.dataloader_num_workers,
    )

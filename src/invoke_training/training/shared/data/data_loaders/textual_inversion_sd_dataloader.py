import typing

from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from invoke_training.training.config.data_config import TextualInversionDataLoaderConfig
from invoke_training.training.shared.data.datasets.image_dir_dataset import (
    ImageDirDataset,
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
from invoke_training.training.shared.data.transforms.textual_inversion_caption_transform import (
    TextualInversionCaptionTransform,
)


def _get_preset_ti_caption_templates(learnable_property: typing.Literal["object", "style"]) -> list[str]:
    if learnable_property == "object":
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
    elif learnable_property == "style":
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
            "a cropped painting in the style of {}",
            "a good painting in the style of {}",
            "a close-up painting in the style of {}",
            "a rendition in the style of {}",
            "a nice painting in the style of {}",
            "a small painting in the style of {}",
            "a weird painting in the style of {}",
            "a large painting in the style of {}",
        ]
    else:
        raise ValueError(f"Unrecognized learnable property type: '{learnable_property}'.")


def build_textual_inversion_sd_dataloader(
    config: TextualInversionDataLoaderConfig,
    placeholder_str: str,
    learnable_property: typing.Literal["object", "style"],
    tokenizer: typing.Optional[CLIPTokenizer],
    batch_size: int,
    caption_templates: typing.Optional[list[str]] = None,
    vae_output_cache_dir: typing.Optional[str] = None,
    shuffle: bool = True,
) -> DataLoader:
    """Construct a DataLoader for a Textual Inversion dataset for Stable Diffusion v1/v2..

    Args:
        config (ImageCaptionDataLoaderConfig): The dataset config.
        placeholder_str (str): The placeholder string being trained.
        learnable_property (str): One of ["object", "style"] indicating the type of training being performed.
        tokenizer (CLIPTokenizer, option): The tokenizer to apply to the captions. Can be None if
            `text_encoder_output_cache_dir` is set.
        batch_size (int): The DataLoader batch size.
        vae_output_cache_dir (str, optional): The directory where VAE outputs are cached and should be loaded from. If
            set, then the image augmentation transforms will be skipped, and the image will not be copied to VRAM.
        shuffle (bool, optional): Whether to shuffle the dataset order.
    Returns:
        DataLoader
    """

    base_dataset = ImageDirDataset(image_dir=config.dataset_dir, image_extensions=config.image_file_extensions)

    if caption_templates is None:
        caption_templates = _get_preset_ti_caption_templates(learnable_property)

    all_transforms = [
        TextualInversionCaptionTransform(
            field_name="caption",
            placeholder_str=placeholder_str,
            caption_templates=caption_templates,
        ),
        SDTokenizeTransform(tokenizer),
    ]

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

    dataset = TransformDataset(base_dataset, all_transforms)

    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=config.dataloader_num_workers,
    )

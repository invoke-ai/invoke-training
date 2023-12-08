import math

import torch

from invoke_training.training.config.data_config import ImageTransformConfig
from invoke_training.training.config.finetune_lora_config import ImageCaptionDataLoaderConfig
from invoke_training.training2.shared.data.data_loaders.image_caption_sdxl_dataloader import (
    build_image_caption_sdxl_dataloader,
)


def test_build_image_caption_sdxl_dataloader():
    """Smoke test of build_image_caption_sdxl_dataloader(...)."""

    config = ImageCaptionDataLoaderConfig(
        dataset_name="lambdalabs/pokemon-blip-captions", image_transforms=ImageTransformConfig(resolution=512)
    )
    data_loader = build_image_caption_sdxl_dataloader(config, 4)

    # 833 is the length of the dataset determined manually here:
    # https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions
    assert len(data_loader) == math.ceil(833 / 4)

    example = next(iter(data_loader))
    assert set(example.keys()) == {"image", "id", "caption", "original_size_hw", "crop_top_left_yx"}

    image = example["image"]
    assert image.shape == (4, 3, 512, 512)
    assert image.dtype == torch.float32

    assert len(example["caption"]) == 4

    original_size_hw = example["original_size_hw"]
    assert len(original_size_hw) == 4
    assert len(original_size_hw[0]) == 2

    crop_top_left_yx = example["crop_top_left_yx"]
    assert len(crop_top_left_yx) == 4
    assert len(crop_top_left_yx[0]) == 2

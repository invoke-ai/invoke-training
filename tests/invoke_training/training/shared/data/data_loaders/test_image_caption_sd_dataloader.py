import math

import torch

from invoke_training.config.pipelines.finetune_lora_config import ImageCaptionDataLoaderConfig
from invoke_training.config.shared.data.data_config import ImageTransformConfig
from invoke_training.training.shared.data.data_loaders.image_caption_sd_dataloader import (
    build_image_caption_sd_dataloader,
)


def test_build_image_caption_sd_dataloader():
    """Smoke test of build_image_caption_sd_dataloader(...)."""
    config = ImageCaptionDataLoaderConfig(
        dataset_name="lambdalabs/pokemon-blip-captions", image_transforms=ImageTransformConfig(resolution=512)
    )
    data_loader = build_image_caption_sd_dataloader(config, 4)

    # 833 is the length of the dataset determined manually here:
    # https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions
    assert len(data_loader) == math.ceil(833 / 4)

    example = next(iter(data_loader))
    assert set(example.keys()) == {"image", "caption", "id"}

    image = example["image"]
    assert image.shape == (4, 3, 512, 512)
    assert image.dtype == torch.float32

    assert len(example["caption"]) == 4

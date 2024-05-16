import math

import torch

from invoke_training._shared.data.data_loaders.image_caption_sd_dataloader import build_image_caption_sd_dataloader
from invoke_training.config.data.data_loader_config import ImageCaptionSDDataLoaderConfig
from invoke_training.config.data.dataset_config import ImageCaptionJsonlDatasetConfig

from ..dataset_fixtures import image_caption_jsonl  # noqa: F401


def test_build_image_caption_sd_dataloader(image_caption_jsonl):  # noqa: F811
    """Smoke test of build_image_caption_sd_dataloader(...)."""

    config = ImageCaptionSDDataLoaderConfig(
        dataset=ImageCaptionJsonlDatasetConfig(jsonl_path=str(image_caption_jsonl)),
    )
    data_loader = build_image_caption_sd_dataloader(config, 4)

    # The dataset has length 5, so the data loader should have 2 batches.
    assert len(data_loader) == math.ceil(5 / 4)

    example = next(iter(data_loader))
    assert set(example.keys()) == {"image", "mask", "id", "caption", "original_size_hw", "crop_top_left_yx"}

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

import math

import pytest
import torch
from transformers import CLIPTokenizer

from invoke_training.training.config.finetune_lora_config import (
    ImageCaptionDataLoaderConfig,
)
from invoke_training.training.shared.data.data_loaders.image_caption_sdxl_dataloader import (
    build_image_caption_sdxl_dataloader,
)


@pytest.mark.loads_model
def test_build_image_caption_sdxl_dataloader():
    """Smoke test of build_image_caption_sdxl_dataloader(...)."""

    tokenizer_1 = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="tokenizer",
        local_files_only=True,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="tokenizer_2",
        local_files_only=True,
    )

    config = ImageCaptionDataLoaderConfig(dataset_name="lambdalabs/pokemon-blip-captions", resolution=512)
    data_loader = build_image_caption_sdxl_dataloader(config, tokenizer_1, tokenizer_2, 4)

    # 833 is the length of the dataset determined manually here:
    # https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions
    assert len(data_loader) == math.ceil(833 / 4)

    example = next(iter(data_loader))
    assert set(example.keys()) == {
        "image",
        "id",
        "original_size_hw",
        "crop_top_left_yx",
        "caption_token_ids_1",
        "caption_token_ids_2",
    }

    image = example["image"]
    assert image.shape == (4, 3, 512, 512)
    assert image.dtype == torch.float32

    original_size_hw = example["original_size_hw"]
    assert len(original_size_hw) == 4
    assert len(original_size_hw[0]) == 2

    crop_top_left_yx = example["crop_top_left_yx"]
    assert len(crop_top_left_yx) == 4
    assert len(crop_top_left_yx[0]) == 2

    caption_token_ids_1 = example["caption_token_ids_1"]
    assert caption_token_ids_1.shape == (4, 77)
    assert caption_token_ids_1.dtype == torch.int64

    caption_token_ids_2 = example["caption_token_ids_2"]
    assert caption_token_ids_2.shape == (4, 77)
    assert caption_token_ids_2.dtype == torch.int64

import math

import pytest
import torch
from transformers import CLIPTokenizer

from invoke_training.training.config.finetune_lora_config import (
    ImageCaptionDataLoaderConfig,
)
from invoke_training.training.shared.data.data_loaders.image_caption_sd_dataloader import (
    build_image_caption_sd_dataloader,
)


@pytest.mark.loads_model
def test_build_image_caption_sd_dataloader():
    """Smoke test of build_image_caption_sd_dataloader(...)."""

    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer",
        local_files_only=True,
        revision="c9ab35ff5f2c362e9e22fbafe278077e196057f0",
    )

    config = ImageCaptionDataLoaderConfig(dataset_name="lambdalabs/pokemon-blip-captions", resolution=512)
    data_loader = build_image_caption_sd_dataloader(config, tokenizer, 4)

    # 833 is the length of the dataset determined manually here:
    # https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions
    assert len(data_loader) == math.ceil(833 / 4)

    example = next(iter(data_loader))
    assert set(example.keys()) == {"image", "caption", "id", "caption_token_ids"}

    image = example["image"]
    assert image.shape == (4, 3, 512, 512)
    assert image.dtype == torch.float32

    caption_token_ids = example["caption_token_ids"]
    assert caption_token_ids.shape == (4, 77)
    assert caption_token_ids.dtype == torch.int64

import pytest
import torch
from transformers import CLIPTokenizer

from invoke_training.training.config.data_config import DreamBoothDataLoaderConfig
from invoke_training.training.shared.data.data_loaders.dreambooth_sd_dataloader import (
    build_dreambooth_sd_dataloader,
)

from ..image_dir_fixture import image_dir  # noqa: F401


@pytest.mark.loads_model
def test_build_dreambooth_sd_dataloader(image_dir):  # noqa: F811
    """Smoke test of build_dreambooth_sd_dataloader(...)."""

    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer",
        local_files_only=True,
        revision="c9ab35ff5f2c362e9e22fbafe278077e196057f0",
    )

    config = DreamBoothDataLoaderConfig(
        instance_prompt="test instance prompt",
        instance_data_dir=str(image_dir),
        class_prompt="test class prompt",
        # For testing, we just use the same directory for the instance and class datasets.
        class_data_dir=str(image_dir),
    )

    data_loader = build_dreambooth_sd_dataloader(
        data_loader_config=config,
        tokenizer=tokenizer,
        batch_size=2,
    )

    assert len(data_loader) == 5  # (5 class images + 5 instance images) / batch size 2

    example = next(iter(data_loader))
    assert set(example.keys()) == {"image", "caption", "id", "caption_token_ids", "loss_weight"}

    image = example["image"]
    assert image.shape == (2, 3, 512, 512)
    assert image.dtype == torch.float32

    caption_token_ids = example["caption_token_ids"]
    assert caption_token_ids.shape == (2, 77)
    assert caption_token_ids.dtype == torch.int64

    loss_weight = example["loss_weight"]
    assert loss_weight.shape == (2,)
    assert loss_weight.dtype == torch.float64


@pytest.mark.loads_model
def test_build_dreambooth_sd_dataloader_no_class_dataset(image_dir):  # noqa: F811
    """Smoke test of build_dreambooth_sd_dataloader(...) without a class dataset."""

    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer",
        local_files_only=True,
        revision="c9ab35ff5f2c362e9e22fbafe278077e196057f0",
    )

    config = DreamBoothDataLoaderConfig(
        instance_prompt="test instance prompt",
        instance_data_dir=str(image_dir),
    )

    data_loader = build_dreambooth_sd_dataloader(
        data_loader_config=config,
        tokenizer=tokenizer,
        batch_size=2,
    )

    assert len(data_loader) == 3  # 5 instance images, batch size 2

    example = next(iter(data_loader))
    assert set(example.keys()) == {"image", "caption", "id", "caption_token_ids", "loss_weight"}

    image = example["image"]
    assert image.shape == (2, 3, 512, 512)
    assert image.dtype == torch.float32

    caption_token_ids = example["caption_token_ids"]
    assert caption_token_ids.shape == (2, 77)
    assert caption_token_ids.dtype == torch.int64

    loss_weight = example["loss_weight"]
    assert loss_weight.shape == (2,)
    assert loss_weight.dtype == torch.float64

import pytest
import torch
from transformers import CLIPTokenizer

from invoke_training.training.config.data_config import DreamBoothDataLoaderConfig
from invoke_training.training.shared.data.data_loaders.dreambooth_sdxl_dataloader import (
    build_dreambooth_sdxl_dataloader,
)

from ..image_dir_fixture import image_dir  # noqa: F401


@pytest.mark.loads_model
def test_build_image_caption_sdxl_dataloader(image_dir):  # noqa: F811
    """Smoke test of build_dreambooth_sdxl_dataloader(...)."""

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

    config = DreamBoothDataLoaderConfig(
        instance_prompt="test instance prompt",
        instance_data_dir=str(image_dir),
        class_prompt="test class prompt",
        # For testing, we just use the same directory for the instance and class datasets.
        class_data_dir=str(image_dir),
    )
    data_loader = build_dreambooth_sdxl_dataloader(
        data_loader_config=config,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        batch_size=2,
    )

    assert len(data_loader) == 5  # (5 class images + 5 instance images) / batch size 2

    example = next(iter(data_loader))
    assert set(example.keys()) == {
        "image",
        "id",
        "original_size_hw",
        "crop_top_left_yx",
        "caption_token_ids_1",
        "caption_token_ids_2",
        "loss_weight",
    }

    image = example["image"]
    assert image.shape == (2, 3, 512, 512)
    assert image.dtype == torch.float32

    original_size_hw = example["original_size_hw"]
    assert len(original_size_hw) == 2
    assert len(original_size_hw[0]) == 2

    crop_top_left_yx = example["crop_top_left_yx"]
    assert len(crop_top_left_yx) == 2
    assert len(crop_top_left_yx[0]) == 2

    caption_token_ids_1 = example["caption_token_ids_1"]
    assert caption_token_ids_1.shape == (2, 77)
    assert caption_token_ids_1.dtype == torch.int64

    caption_token_ids_2 = example["caption_token_ids_2"]
    assert caption_token_ids_2.shape == (2, 77)
    assert caption_token_ids_2.dtype == torch.int64

    loss_weight = example["loss_weight"]
    assert loss_weight.shape == (2,)
    assert loss_weight.dtype == torch.float32


@pytest.mark.loads_model
def test_build_image_caption_sdxl_dataloader_no_class_dataset(image_dir):  # noqa: F811
    """Smoke test of build_dreambooth_sdxl_dataloader(...) without a class dataset."""

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

    config = DreamBoothDataLoaderConfig(
        instance_prompt="test instance prompt",
        instance_data_dir=str(image_dir),
    )
    data_loader = build_dreambooth_sdxl_dataloader(
        data_loader_config=config,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        batch_size=2,
    )

    assert len(data_loader) == 3  # 5 instance images, batch size 2

    example = next(iter(data_loader))
    assert set(example.keys()) == {
        "image",
        "id",
        "original_size_hw",
        "crop_top_left_yx",
        "caption_token_ids_1",
        "caption_token_ids_2",
        "loss_weight",
    }

    image = example["image"]
    assert image.shape == (2, 3, 512, 512)
    assert image.dtype == torch.float32

    original_size_hw = example["original_size_hw"]
    assert len(original_size_hw) == 2
    assert len(original_size_hw[0]) == 2

    crop_top_left_yx = example["crop_top_left_yx"]
    assert len(crop_top_left_yx) == 2
    assert len(crop_top_left_yx[0]) == 2

    caption_token_ids_1 = example["caption_token_ids_1"]
    assert caption_token_ids_1.shape == (2, 77)
    assert caption_token_ids_1.dtype == torch.int64

    caption_token_ids_2 = example["caption_token_ids_2"]
    assert caption_token_ids_2.shape == (2, 77)
    assert caption_token_ids_2.dtype == torch.int64

    loss_weight = example["loss_weight"]
    assert loss_weight.shape == (2,)
    assert loss_weight.dtype == torch.float32

import torch

from invoke_training._shared.data.data_loaders.textual_inversion_sd_dataloader import (
    build_textual_inversion_sd_dataloader,
)
from invoke_training.config.data.data_loader_config import TextualInversionSDDataLoaderConfig
from invoke_training.config.data.dataset_config import ImageCaptionJsonlDatasetConfig, ImageDirDatasetConfig

from ..dataset_fixtures import (
    image_caption_jsonl,  # noqa: F401
    image_dir,  # noqa: F401
)


def test_build_textual_inversion_sd_dataloader(image_dir):  # noqa: F811
    """Smoke test of build_textual_inversion_sd_dataloader(...)."""

    config = TextualInversionSDDataLoaderConfig(
        dataset=ImageDirDatasetConfig(dataset_dir=str(image_dir)), caption_preset="object"
    )

    data_loader = build_textual_inversion_sd_dataloader(
        config=config,
        placeholder_token="placeholder",
        batch_size=2,
    )

    assert len(data_loader) == 3  # ceil(5 images / batch size 2)

    example = next(iter(data_loader))
    assert set(example.keys()) == {"image", "id", "caption", "original_size_hw", "crop_top_left_yx"}

    image = example["image"]
    assert image.shape == (2, 3, 512, 512)
    assert image.dtype == torch.float32

    assert len(example["caption"]) == 2
    for caption in example["caption"]:
        assert "placeholder" in caption

    original_size_hw = example["original_size_hw"]
    assert len(original_size_hw) == 2
    assert len(original_size_hw[0]) == 2

    crop_top_left_yx = example["crop_top_left_yx"]
    assert len(crop_top_left_yx) == 2
    assert len(crop_top_left_yx[0]) == 2


def test_build_textual_inversion_sd_dataloader_keep_original_captions(image_caption_jsonl):  # noqa: F811
    """Test the keep_original_captions=True option."""
    config = TextualInversionSDDataLoaderConfig(
        dataset=ImageCaptionJsonlDatasetConfig(jsonl_path=str(image_caption_jsonl)),
        caption_templates=["{}"],
        keep_original_captions=True,
    )

    data_loader = build_textual_inversion_sd_dataloader(
        config=config,
        placeholder_token="placeholder",
        batch_size=2,
    )

    example = next(iter(data_loader))
    assert set(example.keys()) == {"image", "id", "caption", "original_size_hw", "crop_top_left_yx"}

    assert len(example["caption"]) == 2
    for caption in example["caption"]:
        assert caption.startswith("placeholder ")

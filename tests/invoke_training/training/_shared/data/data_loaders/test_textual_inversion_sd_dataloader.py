import torch

from invoke_training.config.shared.data.data_loader_config import TextualInversionSDDataLoaderConfig
from invoke_training.config.shared.data.dataset_config import ImageDirDatasetConfig
from invoke_training.config.shared.data.transform_config import (
    SDImageTransformConfig,
    TextualInversionPresetCaptionTransformConfig,
)
from invoke_training.training._shared.data.data_loaders.textual_inversion_sd_dataloader import (
    build_textual_inversion_sd_dataloader,
)

from ..image_dir_fixture import image_dir  # noqa: F401


def test_build_textual_inversion_sd_dataloader(image_dir):  # noqa: F811
    """Smoke test of build_textual_inversion_sd_dataloader(...)."""

    config = TextualInversionSDDataLoaderConfig(
        dataset=ImageDirDatasetConfig(dataset_dir=str(image_dir)),
        captions=TextualInversionPresetCaptionTransformConfig(preset="object"),
        image_transforms=SDImageTransformConfig(resolution=512),
    )
    placeholder_tokens = ["placeholder", "placeholder_1"]
    data_loader = build_textual_inversion_sd_dataloader(
        config=config,
        placeholder_tokens=placeholder_tokens,
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
        for placeholder_token in placeholder_tokens:
            assert placeholder_token in caption

    original_size_hw = example["original_size_hw"]
    assert len(original_size_hw) == 2
    assert len(original_size_hw[0]) == 2

    crop_top_left_yx = example["crop_top_left_yx"]
    assert len(crop_top_left_yx) == 2
    assert len(crop_top_left_yx[0]) == 2

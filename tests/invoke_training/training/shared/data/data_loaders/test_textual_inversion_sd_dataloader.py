import torch

from invoke_training.config.shared.data.data_config import ImageTransformConfig, TextualInversionDataLoaderConfig
from invoke_training.training.shared.data.data_loaders.textual_inversion_sd_dataloader import (
    build_textual_inversion_sd_dataloader,
)

from ..image_dir_fixture import image_dir  # noqa: F401


def test_build_textual_inversion_sd_dataloader(image_dir):  # noqa: F811
    """Smoke test of build_textual_inversion_sd_dataloader(...)."""

    config = TextualInversionDataLoaderConfig(
        dataset_dir=str(image_dir), image_transforms=ImageTransformConfig(resolution=512)
    )
    data_loader = build_textual_inversion_sd_dataloader(
        config=config,
        placeholder_str="placeholder",
        learnable_property="style",
        batch_size=2,
    )

    assert len(data_loader) == 3  # ceil(5 images / batch size 2)

    example = next(iter(data_loader))
    assert set(example.keys()) == {"image", "caption", "id"}

    image = example["image"]
    assert image.shape == (2, 3, 512, 512)
    assert image.dtype == torch.float32

    assert len(example["caption"]) == 2

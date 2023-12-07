import torch

from invoke_training.training.config.data_config import DreamBoothDataLoaderConfig
from invoke_training.training.shared.data.data_loaders.dreambooth_sd_dataloader import (
    build_dreambooth_sd_dataloader,
)

from ..image_dir_fixture import image_dir  # noqa: F401


def test_build_dreambooth_sd_dataloader(image_dir):  # noqa: F811
    """Smoke test of build_dreambooth_sd_dataloader(...)."""

    config = DreamBoothDataLoaderConfig(
        instance_prompt="test instance prompt",
        instance_data_dir=str(image_dir),
        class_prompt="test class prompt",
        # For testing, we just use the same directory for the instance and class datasets.
        class_data_dir=str(image_dir),
    )

    data_loader = build_dreambooth_sd_dataloader(data_loader_config=config, batch_size=2)

    assert len(data_loader) == 5  # (5 class images + 5 instance images) / batch size 2

    example = next(iter(data_loader))
    assert set(example.keys()) == {"image", "caption", "id", "loss_weight"}

    image = example["image"]
    assert image.shape == (2, 3, 512, 512)
    assert image.dtype == torch.float32

    caption = example["caption"]
    assert caption == ["test instance prompt", "test class prompt"]

    loss_weight = example["loss_weight"]
    assert loss_weight.shape == (2,)
    assert loss_weight.dtype == torch.float64


def test_build_dreambooth_sd_dataloader_no_class_dataset(image_dir):  # noqa: F811
    """Smoke test of build_dreambooth_sd_dataloader(...) without a class dataset."""
    config = DreamBoothDataLoaderConfig(instance_prompt="test instance prompt", instance_data_dir=str(image_dir))

    data_loader = build_dreambooth_sd_dataloader(data_loader_config=config, batch_size=2)

    assert len(data_loader) == 3  # 5 instance images, batch size 2

    example = next(iter(data_loader))
    assert set(example.keys()) == {"image", "caption", "id", "loss_weight"}

    image = example["image"]
    assert image.shape == (2, 3, 512, 512)
    assert image.dtype == torch.float32

    caption = example["caption"]
    assert caption == ["test instance prompt", "test instance prompt"]

    loss_weight = example["loss_weight"]
    assert loss_weight.shape == (2,)
    assert loss_weight.dtype == torch.float64

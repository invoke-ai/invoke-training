import pytest
import torch
from transformers import CLIPTokenizer

from invoke_training.training.config.data_config import (
    ImageTransformConfig,
    TextualInversionDataLoaderConfig,
)
from invoke_training.training.shared.data.data_loaders.textual_inversion_sd_dataloader import (
    build_textual_inversion_sd_dataloader,
)

from ..image_dir_fixture import image_dir  # noqa: F401


@pytest.mark.loads_model
def test_build_textual_inversion_sd_dataloader(image_dir):  # noqa: F811
    """Smoke test of build_textual_inversion_sd_dataloader(...)."""

    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer",
        local_files_only=True,
        revision="c9ab35ff5f2c362e9e22fbafe278077e196057f0",
    )

    config = TextualInversionDataLoaderConfig(
        dataset_dir=str(image_dir), image_transforms=ImageTransformConfig(resolution=512)
    )
    data_loader = build_textual_inversion_sd_dataloader(
        config=config,
        placeholder_str="placeholder",
        learnable_property="style",
        tokenizer=tokenizer,
        batch_size=2,
    )

    assert len(data_loader) == 3  # ceil(5 images / batch size 2)

    example = next(iter(data_loader))
    assert set(example.keys()) == {"image", "caption", "id", "caption_token_ids"}

    image = example["image"]
    assert image.shape == (2, 3, 512, 512)
    assert image.dtype == torch.float32

    caption_token_ids = example["caption_token_ids"]
    assert caption_token_ids.shape == (2, 77)
    assert caption_token_ids.dtype == torch.int64

import unittest

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import CLIPTokenizer

from invoke_training.training.shared.data.datasets.image_caption_sdxl_dataset import (
    ImageCaptionSDXLDataset,
)


def test_image_caption_sdxl_dataset_len():
    """Test that the ImageCaptionSDXLDataset __len__() function returns the length of the underlying base_dataset."""
    base_dataset_mock = unittest.mock.MagicMock()
    base_dataset_mock.__len__.return_value = 5

    dataset = ImageCaptionSDXLDataset(base_dataset_mock, None, None, resolution=512)

    assert len(dataset) == 5


@pytest.mark.loads_model
def test_image_caption_sdxl_dataset_getitem():
    """Test that the ImageCaptionSDXLDataset __getitem__() function returns a valid example."""
    # Prepare mock base_dataset.
    rgb_np = np.ones((256, 128, 3), dtype=np.uint8)
    rgb_pil = Image.fromarray(rgb_np)
    base_dataset_mock = unittest.mock.MagicMock()
    base_dataset_mock.__getitem__.return_value = {"image": rgb_pil, "caption": "This is a test caption."}

    # Load tokenizers.
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

    # We expect the shape of the 256x128 input image to be transformed as follows:
    # 1. Resize to 1024x512.
    # 2. Center crop at top_left_yx = (256, 0) to produce a 512x512 image.
    dataset = ImageCaptionSDXLDataset(
        base_dataset_mock, tokenizer_1, tokenizer_2, center_crop=True, random_flip=False, resolution=512
    )

    example = dataset[0]

    base_dataset_mock.__getitem__.assert_called_with(0)
    assert set(example.keys()) == {
        "image",
        "original_size_hw",
        "crop_top_left_yx",
        "caption_token_ids_1",
        "caption_token_ids_2",
    }

    image = example["image"]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 512, 512)
    assert image.dtype == torch.float32

    caption_token_ids = example["caption_token_ids_1"]
    assert isinstance(caption_token_ids, torch.Tensor)
    assert caption_token_ids.shape == (77,)
    assert caption_token_ids.dtype == torch.int64

    caption_token_ids = example["caption_token_ids_2"]
    assert isinstance(caption_token_ids, torch.Tensor)
    assert caption_token_ids.shape == (77,)
    assert caption_token_ids.dtype == torch.int64

    original_size_hw = example["original_size_hw"]
    assert isinstance(original_size_hw, tuple)
    assert original_size_hw == (256, 128)

    crop_top_left_yx = example["crop_top_left_yx"]
    assert isinstance(crop_top_left_yx, tuple)
    assert crop_top_left_yx == (256, 0)

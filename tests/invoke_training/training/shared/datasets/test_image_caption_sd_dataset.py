import unittest

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import CLIPTokenizer

from invoke_training.training.shared.datasets.image_caption_sd_dataset import (
    ImageCaptionSDDataset,
)


def test_image_caption_dataset_len():
    """Test that the ImageCaptionSDDataset __len__() function returns the length of the underlying reader."""
    reader_mock = unittest.mock.MagicMock()
    reader_mock.__len__.return_value = 5

    dataset = ImageCaptionSDDataset(reader_mock, None, resolution=512)

    assert len(dataset) == 5


@pytest.mark.loads_model
def test_image_caption_dataset_getitem():
    """Test that the ImageCaptionSDDataset __getitem__() function returns an example with the expected type and
    dimensions.
    """
    # Prepare mock reader.
    rgb_np = np.ones((128, 128, 3), dtype=np.uint8)
    rgb_pil = Image.fromarray(rgb_np)
    reader_mock = unittest.mock.MagicMock()
    reader_mock.__getitem__.return_value = {"image": rgb_pil, "caption": "This is a test caption."}

    # Load CLIPTokenizer.
    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer",
        local_files_only=True,
        revision="c9ab35ff5f2c362e9e22fbafe278077e196057f0",
    )

    dataset = ImageCaptionSDDataset(reader_mock, tokenizer, resolution=512)

    example = dataset[0]

    reader_mock.__getitem__.assert_called_with(0)
    assert set(example.keys()) == {"image", "caption_token_ids"}

    image = example["image"]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 512, 512)
    assert image.dtype == torch.float32

    caption_token_ids = example["caption_token_ids"]
    assert isinstance(caption_token_ids, torch.Tensor)
    assert caption_token_ids.shape == (77,)
    assert caption_token_ids.dtype == torch.int64

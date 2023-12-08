import numpy as np
import pytest
import torch
from PIL import Image

from invoke_training.training2.shared.data.transforms.sd_image_transform import SDImageTransform


@pytest.mark.parametrize("center_crop", [True, False])
def test_sd_image_transform_resolution(center_crop: bool):
    """Test that SDImageTransform resizes and crops to the correct resolution."""
    in_image_np = np.ones((256, 128, 3), dtype=np.uint8)
    in_image_pil = Image.fromarray(in_image_np)

    resolution = 512
    tf = SDImageTransform(resolution, center_crop=center_crop)

    out_example = tf({"image": in_image_pil})

    out_image = out_example["image"]

    assert isinstance(out_image, torch.Tensor)
    assert out_image.shape == (3, resolution, resolution)


def test_sd_image_transform_range():
    """Test that SDImageTransform normalizes images to the range [-1.0, 1.0]."""
    resolution = 128
    in_image_np = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    in_image_np[0, 0, :] = 255  # Image contains one pixel with value 255, and the rest are zeros.
    in_image_pil = Image.fromarray(in_image_np)

    tf = SDImageTransform(resolution)

    out_example = tf({"image": in_image_pil})

    out_image = out_example["image"]
    out_np = np.array(out_image)

    assert np.allclose(out_np[:, 0, 0], 1.0)
    assert np.allclose(out_np[:, 1:, 1:], -1.0)

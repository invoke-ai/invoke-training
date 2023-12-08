import unittest.mock

import numpy as np
import torch
from PIL import Image

from invoke_training.training.shared.data.transforms.sdxl_image_transform import SDXLImageTransform


def denormalize_image(img: np.ndarray) -> np.ndarray:
    """Convert a normalized CxHxW image in range [-1.0, 1.0] to a HxWxC image in the range [0, 255].

    Args:
        img (np.ndarray): Image to denormalize.

    Returns:
        np.ndarray: Result image.
    """
    # Convert back to range [0, 1.0].
    img = img * 0.5 + 0.5
    # Convert back to range [0, 255].
    img *= 255
    # Move channel axis from first dimension to last dimension.
    img = np.moveaxis(img, 0, -1)

    return img


def test_sdxl_image_transform_resolution():
    """Test that SDXLImageTransform resizes and crops to the target resolution, and correctly sets original_size_hw."""
    in_image_np = np.ones((256, 128, 3), dtype=np.uint8)
    in_image_pil = Image.fromarray(in_image_np)

    resolution = 512
    tf = SDXLImageTransform(resolution=resolution)

    out_example = tf({"image": in_image_pil})

    out_image = out_example["image"]
    assert isinstance(out_image, torch.Tensor)
    assert out_image.shape == (3, resolution, resolution)

    original_size_hw = out_example["original_size_hw"]
    assert original_size_hw == (256, 128)


def test_sdxl_image_transform_range():
    """Test that SDXLImageTransform normalizes the image to the range [-1.0, 1.0]."""
    resolution = 128
    in_image_np = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    in_image_np[0, 0, :] = 255  # Image contains one pixel with value 255, and the rest are zeros.
    in_image_pil = Image.fromarray(in_image_np)

    tf = SDXLImageTransform(resolution=resolution)

    out_example = tf({"image": in_image_pil})

    out_image = out_example["image"]
    out_np = np.array(out_image)

    assert np.allclose(out_np[:, 0, 0], 1.0)
    assert np.allclose(out_np[:, 1:, 1:], -1.0)


def test_sdxl_image_transform_center_crop():
    """Test SDXLImageTransform center cropping."""
    # Input image is 9 x 5.
    in_image_np = np.arange(9 * 5 * 3, dtype=np.uint8).reshape((9, 5, 3))
    in_image_pil = Image.fromarray(np.copy(in_image_np))

    # The target resolution is 5x5 (with center cropping).
    tf = SDXLImageTransform(resolution=5, center_crop=True)

    out_example = tf({"image": in_image_pil})

    out_image = out_example["image"]
    out_image_np = np.array(out_image)

    # Verify that the correct region of the image was cropped.
    assert np.allclose(denormalize_image(out_image_np), in_image_np[2:-2, :, :])
    assert out_example["crop_top_left_yx"] == (2, 0)


def test_sdxl_image_transform_random_crop():
    """Test SDXLImageTransform random cropping."""
    # Input image is 9 x 5.
    in_image_np = np.arange(9 * 5 * 3, dtype=np.uint8).reshape((9, 5, 3))
    in_image_pil = Image.fromarray(np.copy(in_image_np))

    # The target resolution is 5x5 (with random cropping).
    resolution = 5
    tf = SDXLImageTransform(resolution=resolution, center_crop=False)

    out_example = tf({"image": in_image_pil})

    out_image = out_example["image"]
    out_image_np = np.array(out_image)

    # Verify that the crop_top_left_yx value is correct.
    crop_y, crop_x = out_example["crop_top_left_yx"]
    assert np.allclose(
        denormalize_image(out_image_np), in_image_np[crop_y : crop_y + resolution, crop_x : crop_x + resolution, :]
    )


def test_sdxl_image_transform_center_crop_flip():
    """Test SDXLImageTransform center cropping with a horizontal flip."""
    # Input image is 5 x 9.
    in_image_np = np.arange(5 * 9 * 3, dtype=np.uint8).reshape((5, 9, 3))
    in_image_pil = Image.fromarray(np.copy(in_image_np))

    # The target resolution is 5x5 (with center cropping and horizontal flipping).
    tf = SDXLImageTransform(resolution=5, center_crop=True, random_flip=True)

    # Note: We patch random.random() to force a horizontal flip to be applied.
    with unittest.mock.patch("random.random", return_value=0.0):
        out_example = tf({"image": in_image_pil})

    out_image = out_example["image"]
    out_image_np = np.array(out_image)

    # Verify that the correct region of the image was cropped/flipped.
    # For this comparison, we flip the in_image_np first, then apply the expected crop.
    assert np.allclose(denormalize_image(out_image_np), in_image_np[:, ::-1, :][:, 2:-2, :])
    assert out_example["crop_top_left_yx"] == (0, 2)


def test_sdxl_image_transform_random_crop_flip():
    """Test SDXLImageTransform random cropping with a horizontal flip."""
    # Input image is 5 x 9.
    in_image_np = np.arange(5 * 9 * 3, dtype=np.uint8).reshape((5, 9, 3))
    in_image_pil = Image.fromarray(np.copy(in_image_np))

    # The target resolution is 5x5 (with random cropping and horizontal flipping).
    resolution = 5
    tf = SDXLImageTransform(resolution=resolution, center_crop=False, random_flip=True)

    # Note: We patch random.random() to force a horizontal flip to be applied.
    with unittest.mock.patch("random.random", return_value=0.0):
        out_example = tf({"image": in_image_pil})

    out_image = out_example["image"]
    out_image_np = np.array(out_image)

    # Verify that the crop_top_left_yx value is correct.
    # For this comparison, we flip the in_image_np first, then apply the expected crop.
    crop_y, crop_x = out_example["crop_top_left_yx"]
    assert np.allclose(
        denormalize_image(out_image_np),
        in_image_np[:, ::-1, :][crop_y : crop_y + resolution, crop_x : crop_x + resolution, :],
    )

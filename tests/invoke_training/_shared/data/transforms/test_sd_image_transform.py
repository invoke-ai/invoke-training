import unittest.mock

import numpy as np
import pytest
import torch
from PIL import Image

from invoke_training._shared.data.transforms.sd_image_transform import SDImageTransform
from invoke_training._shared.data.utils.aspect_ratio_bucket_manager import AspectRatioBucketManager
from invoke_training._shared.data.utils.resolution import Resolution


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


def denormalize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a normalized CxHxW mask in range [0.0, 1.0] to a HxW mask in the range [0, 255]."""
    # Convert back to range [0, 255].
    mask *= 255
    # Squeeze the channel dimension.
    mask = mask.squeeze(0)
    return mask


def test_sd_image_transform_resolution():
    """Test that SDImageTransform resizes and crops to the target resolution, and correctly sets original_size_hw."""
    in_image_np = np.ones((256, 128, 3), dtype=np.uint8)
    in_image_pil = Image.fromarray(in_image_np)
    in_mask_np = np.ones((256, 128), dtype=np.uint8)
    in_mask_pil = Image.fromarray(in_mask_np)

    resolution = Resolution(768, 512)
    tf = SDImageTransform(resolution=resolution)

    out_example = tf({"image": in_image_pil, "mask": in_mask_pil})

    out_image = out_example["image"]
    assert isinstance(out_image, torch.Tensor)
    assert out_image.shape == (3, resolution.height, resolution.width)

    out_mask = out_example["mask"]
    assert isinstance(out_mask, torch.Tensor)
    assert out_mask.shape == (1, resolution.height, resolution.width)

    original_size_hw = out_example["original_size_hw"]
    assert original_size_hw == (256, 128)


def test_sd_image_transform_without_mask():
    """Test that SDImageTransform works correctly when no mask is provided."""
    in_image_np = np.ones((256, 128, 3), dtype=np.uint8)
    in_image_pil = Image.fromarray(in_image_np)

    resolution = Resolution(768, 512)
    tf = SDImageTransform(resolution=resolution)

    # No mask is provided.
    out_example = tf({"image": in_image_pil})

    out_image = out_example["image"]
    assert isinstance(out_image, torch.Tensor)
    assert out_image.shape == (3, resolution.height, resolution.width)

    original_size_hw = out_example["original_size_hw"]
    assert original_size_hw == (256, 128)


def test_sd_image_transform_range():
    """Test that SDImageTransform normalizes the image to the range [-1.0, 1.0], and the mask to the range
    [0.0, 1.0].
    """
    resolution = 128
    in_image_np = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    in_image_np[0, 0, :] = 255  # Image contains one pixel with value 255, and the rest are zeros.
    in_image_pil = Image.fromarray(in_image_np)

    in_mask_np = np.zeros((resolution, resolution), dtype=np.uint8)
    in_mask_np[0, 0] = 255  # Mask contains one pixel with value 255, and the rest are zeros.
    in_mask_pil = Image.fromarray(in_mask_np)

    tf = SDImageTransform(resolution=resolution)

    out_example = tf({"image": in_image_pil, "mask": in_mask_pil})

    out_image = out_example["image"]
    out_np = np.array(out_image)
    assert np.allclose(out_np[:, 0, 0], 1.0)
    assert np.allclose(out_np[:, 1:, 1:], -1.0)

    out_mask = out_example["mask"]
    out_np = np.array(out_mask)
    assert np.allclose(out_np[0, 0, 0], 1.0)
    assert np.allclose(out_np[0, 1:, 1:], 0.0)


def test_sd_image_transform_center_crop():
    """Test SDImageTransform center cropping."""
    # Input image is 9 x 5.
    in_image_np = np.arange(9 * 5 * 3, dtype=np.uint8).reshape((9, 5, 3))
    in_image_pil = Image.fromarray(np.copy(in_image_np))

    mask_image_np = np.arange(9 * 5, dtype=np.uint8).reshape((9, 5))
    mask_image_pil = Image.fromarray(np.copy(mask_image_np))

    # The target resolution is 3x5 (with center cropping).
    tf = SDImageTransform(resolution=(3, 5), center_crop=True)

    out_example = tf({"image": in_image_pil, "mask": mask_image_pil})

    # Verify that the correct region of the image was cropped.
    out_image = out_example["image"]
    out_image_np = np.array(out_image)
    assert np.allclose(denormalize_image(out_image_np), in_image_np[3:-3, :, :])
    assert out_example["crop_top_left_yx"] == (3, 0)

    # Verify that the correct region of the mask was cropped.
    out_mask = out_example["mask"]
    out_mask_np = np.array(out_mask)
    assert np.allclose(denormalize_mask(out_mask_np), mask_image_np[3:-3, :])


def test_sd_image_transform_random_crop():
    """Test SDImageTransform random cropping."""
    # Input image is 9 x 5.
    in_image_np = np.arange(9 * 5 * 3, dtype=np.uint8).reshape((9, 5, 3))
    in_image_pil = Image.fromarray(np.copy(in_image_np))

    mask_image_np = np.arange(9 * 5, dtype=np.uint8).reshape((9, 5))
    mask_image_pil = Image.fromarray(np.copy(mask_image_np))

    # The target resolution is 3x5 (with random cropping).
    resolution = Resolution(3, 5)
    tf = SDImageTransform(resolution=resolution, center_crop=False)

    out_example = tf({"image": in_image_pil, "mask": mask_image_pil})

    # Verify that the crop_top_left_yx value is correct.
    out_image = out_example["image"]
    out_image_np = np.array(out_image)
    crop_y, crop_x = out_example["crop_top_left_yx"]
    assert np.allclose(
        denormalize_image(out_image_np),
        in_image_np[crop_y : crop_y + resolution.height, crop_x : crop_x + resolution.width, :],
    )

    # Verify that the mask was cropped in the same way as the image.
    out_mask = out_example["mask"]
    out_mask_np = np.array(out_mask)
    assert np.allclose(
        denormalize_mask(out_mask_np),
        mask_image_np[crop_y : crop_y + resolution.height, crop_x : crop_x + resolution.width],
    )


def test_sd_image_transform_center_crop_flip():
    """Test SDImageTransform center cropping with a horizontal flip."""
    # Input image is 5 x 9.
    in_image_np = np.arange(5 * 9 * 3, dtype=np.uint8).reshape((5, 9, 3))
    in_image_pil = Image.fromarray(np.copy(in_image_np))

    in_mask_np = np.arange(5 * 9, dtype=np.uint8).reshape((5, 9))
    in_mask_pil = Image.fromarray(np.copy(in_mask_np))

    # The target resolution is 5x3 (with center cropping and horizontal flipping).
    tf = SDImageTransform(resolution=Resolution(5, 3), center_crop=True, random_flip=True)

    # Note: We patch random.random() to force a horizontal flip to be applied.
    with unittest.mock.patch("random.random", return_value=0.0):
        out_example = tf({"image": in_image_pil, "mask": in_mask_pil})

    # Verify that the correct region of the image was cropped/flipped.
    # For this comparison, we flip the in_image_np first, then apply the expected crop.
    out_image = out_example["image"]
    out_image_np = np.array(out_image)
    assert np.allclose(denormalize_image(out_image_np), in_image_np[:, ::-1, :][:, 3:-3, :])
    assert out_example["crop_top_left_yx"] == (0, 3)

    # Verify that the correct region of the mask was cropped/flipped.
    out_mask = out_example["mask"]
    out_mask_np = np.array(out_mask)
    assert np.allclose(denormalize_mask(out_mask_np), in_mask_np[:, ::-1][:, 3:-3])


def test_sd_image_transform_random_crop_flip():
    """Test SDImageTransform random cropping with a horizontal flip."""
    # Input image is 5 x 9.
    in_image_np = np.arange(5 * 9 * 3, dtype=np.uint8).reshape((5, 9, 3))
    in_image_pil = Image.fromarray(np.copy(in_image_np))

    in_mask_np = np.arange(5 * 9, dtype=np.uint8).reshape((5, 9))
    in_mask_pil = Image.fromarray(np.copy(in_mask_np))

    # The target resolution is 5x3 (with random cropping and horizontal flipping).
    resolution = Resolution(5, 3)
    tf = SDImageTransform(resolution=resolution, center_crop=False, random_flip=True)

    # Note: We patch random.random() to force a horizontal flip to be applied.
    with unittest.mock.patch("random.random", return_value=0.0):
        out_example = tf({"image": in_image_pil, "mask": in_mask_pil})

    # Verify that the crop_top_left_yx value is correct.
    # For this comparison, we flip the in_image_np first, then apply the expected crop.
    out_image = out_example["image"]
    out_image_np = np.array(out_image)
    crop_y, crop_x = out_example["crop_top_left_yx"]
    assert np.allclose(
        denormalize_image(out_image_np),
        in_image_np[:, ::-1, :][crop_y : crop_y + resolution.height, crop_x : crop_x + resolution.width, :],
    )

    # Verify thath the mask was cropped in the same way as the image.
    out_mask = out_example["mask"]
    out_mask_np = np.array(out_mask)
    assert np.allclose(
        denormalize_mask(out_mask_np),
        in_mask_np[:, ::-1][crop_y : crop_y + resolution.height, crop_x : crop_x + resolution.width],
    )


def test_sd_image_transform_aspect_ratio_bucket_manager():
    # Input image is 9 x 5.
    in_image_np = np.arange(9 * 5 * 3, dtype=np.uint8).reshape((9, 5, 3))
    in_image_pil = Image.fromarray(np.copy(in_image_np))

    in_mask_np = np.arange(9 * 5, dtype=np.uint8).reshape((9, 5))
    in_mask_pil = Image.fromarray(np.copy(in_mask_np))

    # Initialize SDImageTransform with an AspectRatioBucketManager that has a single 3x5 bucket.
    aspect_ratio_bucket_manager = AspectRatioBucketManager(buckets={Resolution(3, 5)})
    tf = SDImageTransform(resolution=None, aspect_ratio_bucket_manager=aspect_ratio_bucket_manager, center_crop=True)

    out_example = tf({"image": in_image_pil, "mask": in_mask_pil})

    # Verify that the correct region of the image was cropped.
    out_image = out_example["image"]
    out_image_np = np.array(out_image)
    assert np.allclose(denormalize_image(out_image_np), in_image_np[3:-3, :, :])
    assert out_example["crop_top_left_yx"] == (3, 0)

    # Verify that the correct region of the mask was cropped.
    out_mask = out_example["mask"]
    out_mask_np = np.array(out_mask)
    assert np.allclose(denormalize_mask(out_mask_np), in_mask_np[3:-3, :])


@pytest.mark.parametrize(
    ["resolution", "aspect_ratio_bucket_manager"],
    [
        (Resolution(512, 512), AspectRatioBucketManager({})),
        (None, None),
    ],
)
def test_sd_image_transform_resolution_input_validation(
    resolution: Resolution | None, aspect_ratio_bucket_manager: AspectRatioBucketManager | None
):
    with pytest.raises(ValueError):
        _ = SDImageTransform(resolution=resolution, aspect_ratio_bucket_manager=aspect_ratio_bucket_manager)

import math

from PIL.Image import Image
from torchvision import transforms

from invoke_training.training._shared.data.utils.resolution import Resolution


def resize_to_cover(image: Image, size_to_cover: Resolution) -> Image:
    """Resize image to the smallest size that covers 'size_to_cover' while preserving its aspect ratio.

    In other words, achieve the following:
    - resized_height >= size_to_cover.height
    - resized_width >= size_to_cover.width
    - resized_height == size_to_cover.height or resized_width == size_to_cover.width
    - 'image' aspect ratio is preserved.
    """

    scale_to_height = size_to_cover.height / image.height
    scale_to_width = size_to_cover.width / image.width

    if scale_to_height > scale_to_width:
        resize_height = size_to_cover.height
        resize_width = math.ceil(image.width * scale_to_height)
    else:
        resize_width = size_to_cover.width
        resize_height = math.ceil(image.height * scale_to_width)

    resize_transform = transforms.Resize(
        (resize_height, resize_width), interpolation=transforms.InterpolationMode.BILINEAR
    )

    return resize_transform(image)

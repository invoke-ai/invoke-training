import random
import typing

from PIL.Image import Image
from torchvision import transforms
from torchvision.transforms.functional import crop


class SDXLImageTransform:
    """A transform that prepares and augments images for Stable Diffusion XL training."""

    def __init__(self, resolution: int, center_crop: bool = False, random_flip: bool = False):
        self._resolution = resolution
        self._resize_transform = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        self._center_crop_enabled = center_crop
        self._crop_transform = transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
        self._random_flip_enabled = random_flip
        self._flip_transform = transforms.RandomHorizontalFlip(p=1.0)
        self._other_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                # Convert pixel values from range [0, 1.0] to range [-1.0, 1.0]. Normalize applies the following
                # transform: out = (in - 0.5) / 0.5
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __call__(self, data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        # This SDXL image pre-processing logic is adapted from:
        # https://github.com/huggingface/diffusers/blob/7b07f9812a58bfa96c06ed8ffe9e6b584286e2fd/examples/text_to_image/train_text_to_image_lora_sdxl.py#L850-L873

        image: Image = data["image"]

        original_size_hw = (image.height, image.width)

        # Resize smaller image dimension to `resolution`.
        image = self._resize_transform(image)

        # Apply cropping, and record top left crop position.
        if self._center_crop_enabled:
            top_left_y = max(0, int(round((image.height - self._resolution) / 2.0)))
            top_left_x = max(0, int(round((image.width - self._resolution) / 2.0)))
            image = self._crop_transform(image)
        else:
            top_left_y, top_left_x, h, w = self._crop_transform.get_params(image, (self._resolution, self._resolution))
            image = crop(image, top_left_y, top_left_x, h, w)

        # Apply random flip and update top left crop position accordingly.
        if self._random_flip_enabled and random.random() < 0.5:
            top_left_x = original_size_hw[1] - image.width - top_left_x
            image = self._flip_transform(image)

        crop_top_left_yx = (top_left_y, top_left_x)

        # Convert image to Tensor and normalize to range [-1.0, 1.0].
        image = self._other_transforms(image)

        data["image"] = image
        data["original_size_hw"] = original_size_hw
        data["crop_top_left_yx"] = crop_top_left_yx
        return data

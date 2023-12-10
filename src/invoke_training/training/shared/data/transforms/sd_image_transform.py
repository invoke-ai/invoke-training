import typing

from torchvision import transforms


class SDImageTransform:
    """A transform that prepares and augments images for Stable Diffusion v1/v2 training."""

    def __init__(self, resolution: int, center_crop: bool = True, random_flip: bool = False):
        """Initialize SDImageTransform.

        Args:
            resolution (int): The image resolution that will be produced (square images are assumed).
            center_crop (bool, optional): If True, crop to the center of the image to achieve the target resolution. If
                False, crop at a random location.
            random_flip (bool, optional): Whether to apply a random horizontal flip to the images.
        """
        self._image_transforms = transforms.Compose(
            [
                # Resize smaller image dimension to `resolution`.
                transforms.Resize(
                    resolution,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                # Crop to `resolution` x `resolution`.
                (transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)),
                (transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x)),
                transforms.ToTensor(),
                # Convert pixel values from range [0, 1.0] to range [-1.0, 1.0]. Normalize applies the following
                # transform: out = (in - 0.5) / 0.5
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __call__(self, data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        data["image"] = self._image_transforms(data["image"])
        return data

import typing

from torchvision import transforms

from invoke_training._shared.data.utils.aspect_ratio_bucket_manager import AspectRatioBucketManager, Resolution
from invoke_training._shared.data.utils.resize import resize_to_cover


class FluxImageTransform:
    """A transform that prepares and augments images for Flux.1-dev training."""

    def __init__(
        self,
        image_field_names: list[str],
        fields_to_normalize_to_range_minus_one_to_one: list[str],
        resolution: int | None = 512,
        aspect_ratio_bucket_manager: AspectRatioBucketManager | None = None,
        random_flip: bool = True,
        orig_size_field_name: str = "original_size_hw",
        crop_field_name: str = "crop_top_left_yx",
    ):
        """Initialize FluxImageTransform.

        Args:
            image_field_names (list[str]): The field names of the images to be transformed.
            resolution (int): The image resolution that will be produced. One of `resolution` and
                `aspect_ratio_bucket_manager` should be non-None.
            aspect_ratio_bucket_manager (AspectRatioBucketManager): The AspectRatioBucketManager used to determine the
                target resolution for each image. One of `resolution` and `aspect_ratio_bucket_manager` should be
                non-None.
            center_crop (bool, optional): If True, crop to the center of the image to achieve the target resolution. If
                False, crop at a random location.
            random_flip (bool, optional): Whether to apply a random horizontal flip to the images.
        """
        self.image_field_names = image_field_names
        self.fields_to_normalize_to_range_minus_one_to_one = fields_to_normalize_to_range_minus_one_to_one
        self.resolution = resolution
        self.aspect_ratio_bucket_manager = aspect_ratio_bucket_manager
        self.random_flip = random_flip

    def __call__(self, data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:  # noqa: C901
        image_fields: dict = {}
        for field_name in self.image_field_names:
            image_fields[field_name] = data[field_name]

        for field_name, image in image_fields.items():
            # Determine the target image resolution.
            if self.resolution is not None:
                resolution = self.resolution
                resolution_obj = Resolution(resolution, resolution)
            else:
                original_size_hw = (image.height, image.width)
                resolution_obj = self.aspect_ratio_bucket_manager.get_aspect_ratio_bucket(
                    Resolution.parse(original_size_hw)
                )

            image = resize_to_cover(image, resolution_obj)
            image = transforms.CenterCrop(resolution)(image)
            image = transforms.ToTensor()(image)

            if self.random_flip:
                image = transforms.RandomHorizontalFlip(p=0.5)(image)
            image_fields[field_name] = image

            if field_name in self.fields_to_normalize_to_range_minus_one_to_one:
                image_fields[field_name] = transforms.Normalize([0.5], [0.5])(image)

        for field_name, image in image_fields.items():
            data[field_name] = image
        return data

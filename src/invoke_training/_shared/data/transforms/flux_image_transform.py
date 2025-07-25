import typing

from torchvision import transforms
from torchvision.transforms.functional import crop

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
        center_crop: bool = True,
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
        self.center_crop = center_crop

    def __call__(self, data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:  # noqa: C901
        image_fields: dict = {}
        for field_name in self.image_field_names:
            image_fields[field_name] = data[field_name]

        # Get the first image to determine original size and resolution
        first_image = next(iter(image_fields.values()))
        original_size_hw = (first_image.height, first_image.width)

        for field_name, image in image_fields.items():
            # Determine the target image resolution.
            if self.resolution is not None:
                resolution = self.resolution
                resolution_obj = Resolution(resolution, resolution)
            else:
                resolution_obj = self.aspect_ratio_bucket_manager.get_aspect_ratio_bucket(
                    Resolution.parse(original_size_hw)
                )

            image = resize_to_cover(image, resolution_obj)
            
            # Apply cropping and record top left crop position
            if self.center_crop:
                top_left_y = max(0, (image.height - resolution_obj.height) // 2)
                top_left_x = max(0, (image.width - resolution_obj.width) // 2)
                image = transforms.CenterCrop(resolution_obj.to_tuple())(image)
            else:
                crop_transform = transforms.RandomCrop(resolution_obj.to_tuple())
                top_left_y, top_left_x, h, w = crop_transform.get_params(image, resolution_obj.to_tuple())
                image = crop(image, top_left_y, top_left_x, resolution_obj.height, resolution_obj.width)

            # Apply random flip and update top left crop position accordingly
            if self.random_flip:
                # TODO: Use a seed for repeatable results
                import random
                if random.random() < 0.5:
                    top_left_x = original_size_hw[1] - image.width - top_left_x
                    image = transforms.RandomHorizontalFlip(p=1.0)(image)

            image = transforms.ToTensor()(image)

            if field_name in self.fields_to_normalize_to_range_minus_one_to_one:
                image_fields[field_name] = transforms.Normalize([0.5], [0.5])(image)
            else:
                image_fields[field_name] = image

        # Store the processed images and metadata
        for field_name, image in image_fields.items():
            data[field_name] = image
        
        # Add metadata fields expected by VAE caching
        data["original_size_hw"] = original_size_hw
        data["crop_top_left_yx"] = (top_left_y, top_left_x)
        
        return data

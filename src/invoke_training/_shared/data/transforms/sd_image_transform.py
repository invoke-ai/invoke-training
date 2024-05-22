import random
import typing

from torchvision import transforms
from torchvision.transforms.functional import crop

from invoke_training._shared.data.utils.aspect_ratio_bucket_manager import AspectRatioBucketManager, Resolution
from invoke_training._shared.data.utils.resize import resize_to_cover


class SDImageTransform:
    """A transform that prepares and augments images for Stable Diffusion training."""

    def __init__(
        self,
        image_field_names: list[str],
        fields_to_normalize_to_range_minus_one_to_one: list[str],
        resolution: int | tuple[int, int] | Resolution | None,
        aspect_ratio_bucket_manager: AspectRatioBucketManager | None = None,
        center_crop: bool = True,
        random_flip: bool = False,
        orig_size_field_name: str = "original_size_hw",
        crop_field_name: str = "crop_top_left_yx",
    ):
        """Initialize SDImageTransform.

        Args:
            image_field_names (list[str]): The field names of the images to be transformed.
            resolution (Resolution): The image resolution that will be produced. One of `resolution` and
                `aspect_ratio_bucket_manager` should be non-None.
            aspect_ratio_bucket_manager (AspectRatioBucketManager): The AspectRatioBucketManager used to determine the
                target resolution for each image. One of `resolution` and `aspect_ratio_bucket_manager` should be
                non-None.
            center_crop (bool, optional): If True, crop to the center of the image to achieve the target resolution. If
                False, crop at a random location.
            random_flip (bool, optional): Whether to apply a random horizontal flip to the images.
        """
        self._image_field_names = image_field_names
        self._fields_to_normalize_to_range_minus_one_to_one = fields_to_normalize_to_range_minus_one_to_one
        if resolution is not None and aspect_ratio_bucket_manager is not None:
            raise ValueError("Only one of `resolution` or `aspect_ratio_bucket_manager` should be set.")

        if resolution is None and aspect_ratio_bucket_manager is None:
            raise ValueError("One of `resolution` or `aspect_ratio_bucket_manager` must be set.")

        self._resolution = Resolution.parse(resolution) if resolution is not None else None
        self._aspect_ratio_bucket_manager = aspect_ratio_bucket_manager
        self._center_crop_enabled = center_crop
        self._random_flip_enabled = random_flip
        self._flip_transform = transforms.RandomHorizontalFlip(p=1.0)
        self._to_tensor_transform = transforms.ToTensor()
        # Convert pixel values from range [0, 1.0] to range [-1.0, 1.0].
        # Normalize applies the following transform: out = (in - 0.5) / 0.5
        self._normalize_image_transform = transforms.Normalize([0.5], [0.5])

        self._orig_size_field_name = orig_size_field_name
        self._crop_field_name = crop_field_name

    def __call__(self, data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:  # noqa: C901
        # This SDXL image pre-processing logic is adapted from:
        # https://github.com/huggingface/diffusers/blob/7b07f9812a58bfa96c06ed8ffe9e6b584286e2fd/examples/text_to_image/train_text_to_image_lora_sdxl.py#L850-L873

        image_fields: dict = {}
        for field_name in self._image_field_names:
            image_fields[field_name] = data[field_name]
        sizes = [image.size for image in image_fields.values()]
        # All images should have the same size.
        assert all(size == sizes[0] for size in sizes)

        # Helper function to access the first image, which is sometimes used to infer the shape of all images.
        def get_first_image():
            return next(iter(image_fields.values()))

        original_size_hw = (get_first_image().height, get_first_image().width)

        # Determine the target image resolution.
        if self._resolution is not None:
            resolution = self._resolution
        else:
            resolution = self._aspect_ratio_bucket_manager.get_aspect_ratio_bucket(Resolution.parse(original_size_hw))

        # Resize to cover the target resolution while preserving aspect ratio.
        for field_name, image in image_fields.items():
            image_fields[field_name] = resize_to_cover(image, resolution)

        # Apply cropping, and record top left crop position.
        if self._center_crop_enabled:
            top_left_y = max(0, (get_first_image().height - resolution.height) // 2)
            top_left_x = max(0, (get_first_image().width - resolution.width) // 2)
        else:
            crop_transform = transforms.RandomCrop(resolution.to_tuple())
            top_left_y, top_left_x, h, w = crop_transform.get_params(get_first_image(), resolution.to_tuple())
        for field_name, image in image_fields.items():
            image_fields[field_name] = crop(image, top_left_y, top_left_x, resolution.height, resolution.width)

        # Apply random flip and update top left crop position accordingly.
        # TODO(ryand): Use a seed for repeatable results.
        if self._random_flip_enabled and random.random() < 0.5:
            top_left_x = original_size_hw[1] - get_first_image().width - top_left_x
            for field_name, image in image_fields.items():
                image_fields[field_name] = self._flip_transform(image)

        crop_top_left_yx = (top_left_y, top_left_x)

        # Convert to Tensors.
        for field_name, image in image_fields.items():
            image_fields[field_name] = self._to_tensor_transform(image)

        # Normalize to range [-1.0, 1.0].
        # HACK(ryand): We should find a better way to determine the normalization range of each image field.
        for field_name, image in image_fields.items():
            if field_name in self._fields_to_normalize_to_range_minus_one_to_one:
                image_fields[field_name] = self._normalize_image_transform(image)

        data[self._orig_size_field_name] = original_size_hw
        data[self._crop_field_name] = crop_top_left_yx
        for field_name, image in image_fields.items():
            data[field_name] = image

        return data

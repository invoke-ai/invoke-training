import random

import PIL
from torchvision import transforms
from torchvision.transforms.functional import crop
from transformers import PreTrainedTokenizer

from invoke_training.training.shared.data.base_image_caption_reader import (
    BaseImageCaptionReader,
)


class ImageCaptionSDXLDataset:
    """A image-caption dataset for Stable Diffusion XL models. This class wraps a BaseImageCaptionReader and applies
    common image transformations and caption tokenization.
    """

    def __init__(
        self,
        reader: BaseImageCaptionReader,
        tokenizer_1: PreTrainedTokenizer,
        tokenizer_2: PreTrainedTokenizer,
        resolution: int,
        center_crop: bool = False,
        random_flip: bool = False,
    ):
        """Initialize ImageCaptionSDDataset.

        Args:
            reader (BaseImageCaptionReader): The reader to wrap.
            tokenizer_1 (PreTrainedTokenizer): The first SDXL text tokenizer.
            tokenizer_2 (PreTrainedTokenizer): The second SDXL text tokenizer.
            resolution (int): The image resolution that will be produced (square images are assumed).
            center_crop (bool, optional): If True, crop to the center of the image to achieve the target resolution. If
                False, crop at a random location.
            random_flip (bool, optional): Whether to apply a random horizontal flip to the images.
        """
        self._reader = reader
        self._tokenizer_1 = tokenizer_1
        self._tokenizer_2 = tokenizer_2

        # Image transforms.
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

    def _tokenize_caption(self, tokenizer, caption: str):
        input = tokenizer(
            caption,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return input.input_ids[0, ...]

    def _preprocess_image(self, image: PIL.Image.Image):
        # This SDXL image pre-processing logic is adapted from:
        # https://github.com/huggingface/diffusers/blob/7b07f9812a58bfa96c06ed8ffe9e6b584286e2fd/examples/text_to_image/train_text_to_image_lora_sdxl.py#L850-L873
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
            top_left_x = image.width - top_left_x
            image = self._flip_transform(image)

        crop_top_left_yx = (top_left_y, top_left_x)

        # Convert image to Tensor and normalize to range [-1.0, 1.0].
        image = self._other_transforms(image)

        return original_size_hw, crop_top_left_yx, image

    def __len__(self) -> int:
        return len(self._reader)

    def __getitem__(self, idx: int):
        example = self._reader[idx]
        original_size_hw, crop_top_left_yx, image = self._preprocess_image(example["image"])
        return {
            "image": image,
            "original_size_hw": original_size_hw,
            "crop_top_left_yx": crop_top_left_yx,
            "caption_token_ids_1": self._tokenize_caption(self._tokenizer_1, example["caption"]),
            "caption_token_ids_2": self._tokenize_caption(self._tokenizer_2, example["caption"]),
        }

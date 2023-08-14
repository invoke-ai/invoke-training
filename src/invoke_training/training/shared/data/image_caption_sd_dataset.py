import torch.utils.data
from torchvision import transforms
from transformers import CLIPTokenizer


class ImageCaptionSDDataset(torch.utils.data.Dataset):
    """A image-caption dataset for Stable Diffusion v1/v2 models. This class wraps a `torch.utils.data.Dataset` and
    applies common image transformations and caption tokenization.
    """

    def __init__(
        self,
        reader: torch.utils.data.Dataset,
        tokenizer: CLIPTokenizer,
        resolution: int,
        center_crop: bool = False,
        random_flip: bool = False,
    ):
        """Initialize ImageCaptionSDDataset.

        Args:
            reader (torch.utils.data.Dataset): The reader to wrap.
            tokenizer (CLIPTokenizer): The tokenizer to apply to the captions.
            resolution (int): The image resolution that will be produced (square images are assumed).
            center_crop (bool, optional): If True, crop to the center of the image to achieve the target resolution. If
                False, crop at a random location.
            random_flip (bool, optional): Whether to apply a random horizontal flip to the images.
        """
        self._reader = reader
        self._tokenizer = tokenizer
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

    def _tokenize_caption(self, caption: str):
        input = self._tokenizer(
            caption,
            max_length=self._tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return input.input_ids[0, ...]

    def __len__(self) -> int:
        return len(self._reader)

    def __getitem__(self, idx: int):
        example = self._reader[idx]
        image = self._image_transforms(example["image"])
        caption_token_ids = self._tokenize_caption(example["caption"])
        return {
            "image": image,
            "caption_token_ids": caption_token_ids,
        }

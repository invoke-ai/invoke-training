import typing
from pathlib import Path

import torch.utils.data
from PIL import Image
from pydantic import BaseModel

from invoke_training._shared.data.utils.resolution import Resolution
from invoke_training._shared.utils.jsonl import load_jsonl, save_jsonl

IMAGE_COLUMN_DEFAULT = "image"
CAPTION_COLUMN_DEFAULT = "text"
MASK_COLUMN_DEFAULT = "mask"


class ImageCaptionExample(BaseModel):
    image_path: str
    mask_path: str | None
    caption: str


class ImageCaptionJsonlDataset(torch.utils.data.Dataset):
    """A dataset that loads images and captions from a directory of image files and .txt files."""

    def __init__(
        self,
        jsonl_path: Path | str,
        image_column: str = IMAGE_COLUMN_DEFAULT,
        caption_column: str = CAPTION_COLUMN_DEFAULT,
        keep_in_memory: bool = False,
    ):
        super().__init__()
        self._jsonl_path = Path(jsonl_path)
        self._image_column = image_column
        self._caption_column = caption_column

        data = load_jsonl(jsonl_path)
        examples: list[ImageCaptionExample] = []
        for d in data:
            # Clear error messages here are helpful in the Gradio UI.
            if image_column not in d:
                raise ValueError(f"Column '{image_column}' not found in jsonl file '{jsonl_path}'.")
            if caption_column not in d:
                raise ValueError(f"Column '{caption_column}' not found in jsonl file '{jsonl_path}'.")
            examples.append(
                ImageCaptionExample(
                    image_path=d[image_column], mask_path=d.get(MASK_COLUMN_DEFAULT, None), caption=d[caption_column]
                )
            )
        self.examples = examples

        self._keep_in_memory = keep_in_memory
        self._example_cache: dict[int, dict[str, typing.Any]] = {}

    def save_jsonl(self):
        data = []
        for example in self.examples:
            data.append({self._image_column: example.image_path, self._caption_column: example.caption})
        save_jsonl(data, self._jsonl_path)

    def _get_image_path(self, idx: int) -> str:
        image_path = self.examples[idx].image_path
        image_path = Path(image_path)

        # image_path could be either absolute, or relative to the jsonl file.
        if not image_path.is_absolute():
            image_path = self._jsonl_path.parent / image_path

        return image_path

    def _get_mask_path(self, idx: int) -> str:
        mask_path = self.examples[idx].mask_path
        mask_path = Path(mask_path)

        # mask_path could be either absolute, or relative to the jsonl file.
        if not mask_path.is_absolute():
            mask_path = self._jsonl_path.parent / mask_path

        return mask_path

    def _load_image(self, image_path: str) -> Image.Image:
        # We call `convert("RGB")` to drop the alpha channel from RGBA images, or to repeat channels for greyscale
        # images.
        return Image.open(image_path).convert("RGB")

    def _load_mask(self, mask_path: str) -> Image.Image:
        return Image.open(mask_path).convert("L")

    def _load_example(self, idx: int) -> dict[str, typing.Any]:
        example = {
            "id": str(idx),
            "image": self._load_image(self._get_image_path(idx)),
            "caption": self.examples[idx].caption,
        }
        if self.examples[idx].mask_path:
            example["mask"] = self._load_mask(self._get_mask_path(idx))
        return example

    def get_image_dimensions(self) -> list[Resolution]:
        """Get the dimensions of all images in the dataset.

        TODO(ryand): Re-think this approach. For large datasets (e.g. streaming from S3) it doesn't make sense to
        calculate this dynamically every time.
        """
        image_dims: list[Resolution] = []
        for i in range(len(self.examples)):
            image = Image.open(self._get_image_path(i))
            image_dims.append(Resolution(image.height, image.width))

        return image_dims

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        if self._keep_in_memory:
            if idx not in self._example_cache:
                self._example_cache[idx] = self._load_example(idx)
            # Return a shallow copy of the example to prevent the caller from modifying the cached example.
            # Shallow rather than deep, because we don't want to copy the image data.
            return self._example_cache[idx].copy()
        return self._load_example(idx)

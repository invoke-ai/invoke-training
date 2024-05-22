import os
import typing

import torch
from PIL import Image


class ImageDirDataset(torch.utils.data.Dataset):
    """A simple dataset that loads images from a directory."""

    def __init__(
        self,
        dataset_dir: str,
        image_extensions: typing.Optional[list[str]] = None,
    ):
        super().__init__()
        if image_extensions is None:
            image_extensions = [".png", ".jpg", ".jpeg"]
        image_extensions = [ext.lower() for ext in image_extensions]

        # Determine the list of image paths to include in the dataset.
        self._image_paths: list[str] = []
        for image_file in os.listdir(dataset_dir):
            image_path = os.path.join(dataset_dir, image_file)
            if os.path.isfile(image_path) and os.path.splitext(image_path)[1].lower() in image_extensions:
                self._image_paths.append(image_path)
        self._image_paths.sort()

    def _load_image(self, image_path: str) -> Image.Image:
        # We call `convert("RGB")` to drop the alpha channel from RGBA images, or to repeat channels for greyscale
        # images.
        return Image.open(image_path).convert("RGB")

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int):
        image_path = self._image_paths[idx]
        image = self._load_image(image_path)
        return {"image_path": self._image_paths[idx], "image": image}


def list_collate_fn(examples):
    """Custom collate_fn that combines images into a list rather than stacking into a tensor. This is what the Moondream
    model expects.
    """
    return {
        "image": [example["image"] for example in examples],
        "image_path": [example["image_path"] for example in examples],
    }

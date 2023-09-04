import os
import typing

import torch.utils.data
from PIL import Image


class ImageDirDataset(torch.utils.data.Dataset):
    """A dataset that loads image files from a directory."""

    def __init__(self, image_dir: str, id_prefix: str = "", image_extensions: typing.Optional[list[str]] = None):
        """Initialize an ImageDirDataset

        Args:
            image_dir (str): The directory to load images from.
            id_prefix (str): A prefix added to the 'id' field in every example.
            image_extensions (list[str], optional): The list of image file extensions to include in the dataset (not
                case-sensitive). Defaults to [".jpg", ".jpeg", ".png"].
        """
        super().__init__()
        self._id_prefix = id_prefix
        if image_extensions is None:
            image_extensions = [".jpg", ".jpeg", ".png"]
        image_extensions = [ext.lower() for ext in image_extensions]

        self._image_paths = []

        for image_file in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_file)
            if os.path.isfile(image_path) and os.path.splitext(image_path)[1].lower() in image_extensions:
                self._image_paths.append(image_path)

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        # We call `convert("RGB")` to drop the alpha channel from RGBA images, or to repeat channels for greyscale
        # images.
        return {"id": f"{self._id_prefix}{idx}", "image": Image.open(self._image_paths[idx]).convert("RGB")}

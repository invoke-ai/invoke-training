import json
import os
import typing

import torch.utils.data
from PIL import Image


class ImagePairPreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: str):
        super().__init__()
        self._dataset_dir = dataset_dir

        # Load metadata from metadata.jsonl.
        self._metadata: list[dict[str, typing.Any]] = []
        with open(os.path.join(self._dataset_dir, "metadata.jsonl")) as f:
            while (line := f.readline()) != "":
                self._metadata.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._metadata)

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        # We call `convert("RGB")` to drop the alpha channel from RGBA images, or to repeat channels for greyscale
        # images.
        example = self._metadata[idx]
        image_0_path = os.path.join(self._dataset_dir, example["image_0"])
        image_1_path = os.path.join(self._dataset_dir, example["image_1"])
        return {
            "id": str(idx),
            "image_0": Image.open(image_0_path).convert("RGB"),
            "image_1": Image.open(image_1_path).convert("RGB"),
            "caption": example["prompt"],
            "prefer_0": example["prefer_0"],
            "prefer_1": example["prefer_1"],
        }

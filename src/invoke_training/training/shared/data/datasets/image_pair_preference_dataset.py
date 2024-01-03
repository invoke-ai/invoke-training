import json
import os
import typing
from pathlib import Path

import torch.utils.data
from PIL import Image


class ImagePairPreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: str):
        super().__init__()
        self._dataset_dir = dataset_dir

        self._metadata = self.load_metadata(self._dataset_dir)

    @classmethod
    def load_metadata(cls, dataset_dir: Path | str) -> list[dict[str, typing.Any]]:
        """Load the dataset metadata from metadata.jsonl."""
        metadata: list[dict[str, typing.Any]] = []
        with open(Path(dataset_dir) / "metadata.jsonl") as f:
            while (line := f.readline()) != "":
                metadata.append(json.loads(line))
        return metadata

    @classmethod
    def save_metadata(
        cls, metadata: list[dict[str, typing.Any]], dataset_dir: str | Path, metadata_file: str = "metadata.jsonl"
    ) -> Path:
        """Load the dataset metadata from metadata.jsonl."""
        metadata_path = Path(dataset_dir) / metadata_file
        with open(metadata_path, "w") as f:
            for m in metadata:
                json.dump(m, f)
                f.write("\n")

        return metadata_path

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

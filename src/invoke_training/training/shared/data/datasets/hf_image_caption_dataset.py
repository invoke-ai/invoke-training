import os
import typing

import datasets
import torch.utils.data
from PIL.Image import Image

from invoke_training.training.shared.data.utils.resolution import Resolution


class HFImageCaptionDataset(torch.utils.data.Dataset):
    """An image-caption dataset wrapper for Hugging Face datasets.

    The wrapped HF dataset can be either from the HF hub, or in Imagefolder format
    (https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder).
    """

    def __init__(self, hf_dataset, image_column: str = "image", caption_column: str = "text"):
        column_names = hf_dataset["train"].column_names
        if image_column not in column_names:
            raise ValueError(
                f"The image_column='{image_column}' is not in the set of dataset column names: '{column_names}'."
            )

        if caption_column not in column_names:
            raise ValueError(
                f"The caption_column='{caption_column}' is not in the set of dataset column names: '{column_names}'."
            )

        self._image_column = image_column

        def preprocess(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            return {
                "image": images,
                "caption": examples[caption_column],
            }

        self._hf_dataset = hf_dataset["train"].with_transform(preprocess)

    @classmethod
    def from_dir(
        cls,
        dataset_dir: str,
        hf_load_dataset_kwargs: typing.Optional[dict[str, typing.Any]] = None,
        image_column: str = "image",
        caption_column: str = "text",
    ):
        """Initialize a HFImageCaptionDataset from a Hugging Face ImageFolder dataset directory
        (https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder).

        Args:
            dataset_dir (str): The path to the dataset directory.
            hf_load_dataset_kwargs (dict[str, typing.Any], optional): kwargs to forward to `datasets.load_dataset(...)`.
            image_column (str, optional): The name of the image column in the dataset. Defaults to "image".
            caption_column (str, optional): The name of the caption column in the dataset. Defaults to "text".
        """
        hf_load_dataset_kwargs = hf_load_dataset_kwargs or {}
        data_files = {"train": os.path.join(dataset_dir, "**")}
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
        hf_dataset = datasets.load_dataset("imagefolder", data_files=data_files, **hf_load_dataset_kwargs)

        return cls(hf_dataset=hf_dataset, image_column=image_column, caption_column=caption_column)

    @classmethod
    def from_hub(
        cls,
        dataset_name: str,
        hf_load_dataset_kwargs: typing.Optional[dict[str, typing.Any]] = None,
        image_column: str = "image",
        caption_column: str = "text",
    ):
        """Initialize a HFImageCaptionDataset from a Hugging Face Hub dataset.

        Args:
            dataset_name (str): The HF Hub dataset name (a.k.a. path).
            hf_load_dataset_kwargs (dict[str, typing.Any], optional): kwargs to forward to `datasets.load_dataset(...)`.
            image_column (str, optional): The name of the image column in the dataset. Defaults to "image".
            caption_column (str, optional): The name of the caption column in the dataset. Defaults to "text".
        """
        hf_load_dataset_kwargs = hf_load_dataset_kwargs or {}
        hf_dataset = datasets.load_dataset(dataset_name, **hf_load_dataset_kwargs)

        return cls(hf_dataset=hf_dataset, image_column=image_column, caption_column=caption_column)

    def get_image_dimensions(self) -> list[Resolution]:
        """Get the dimensions of all images in the dataset.

        TODO(ryand): Re-think this approach. For large datasets (e.g. streaming from S3) it doesn't make sense to
        calculate this dynamically every time.
        """
        image_dims: list[Resolution] = []
        for i in range(len(self._hf_dataset)):
            example = self._hf_dataset[i]
            image: Image = example[self._image_column]
            image_dims.append(Resolution(image.height, image.width))

        return image_dims

    def __len__(self) -> int:
        """Get the dataset length.

        Returns:
            int: The number of image-caption pairs in the dataset.
        """
        return len(self._hf_dataset)

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        """Load the dataset example at index `idx`.

        Raises:
            IndexError: If `idx` is out of range.

        Returns:
            dict: A dataset example with 3 keys: "image", "caption", and "id".
                The "image" key maps to a `PIL` image in RGB format.
                The "caption" key maps to a string.
                The "id" key is the example's index (often used for caching).
        """
        example = self._hf_dataset[idx]
        example["id"] = idx
        return example

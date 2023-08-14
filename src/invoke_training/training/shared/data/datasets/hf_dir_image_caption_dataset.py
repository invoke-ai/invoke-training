import os
import typing

import datasets
import torch.utils.data


class HFDirImageCaptionDataset(torch.utils.data.Dataset):
    """An image-caption dataset for datasets in the Hugging Face Datasets Imagefolder format
    (https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder).
    """

    def __init__(
        self,
        dataset_dir: str,
        hf_load_dataset_kwargs: typing.Optional[dict[str, typing.Any]] = None,
        image_column: str = "image",
        caption_column: str = "text",
    ):
        """Initialize a HFDirImageCaptionDataset.

        Args:
            dataset_dir (str): The path to the dataset directory.
            hf_load_dataset_kwargs (dict[str, typing.Any], optional): kwargs to forward to `datasets.load_dataset(...)`.
            image_column (str, optional): The name of the image column in the dataset. Defaults to "image".
            caption_column (str, optional): The name of the caption column in the dataset. Defaults to "text".

        Raises:
            ValueError: If `image_column` is not found in the dataset.
            ValueError: If `caption_column` is not found in the dataset.
        """
        super().__init__()
        hf_load_dataset_kwargs = hf_load_dataset_kwargs or {}
        data_files = {"train": os.path.join(dataset_dir, "**")}
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
        hf_dataset = datasets.load_dataset("imagefolder", data_files=data_files, **hf_load_dataset_kwargs)

        column_names = hf_dataset["train"].column_names
        if image_column not in column_names:
            raise ValueError(
                f"The image_column='{image_column}' is not in the set of dataset column names: '{column_names}'."
            )

        if caption_column not in column_names:
            raise ValueError(
                f"The caption_column='{caption_column}' is not in the set of dataset column names: '{column_names}'."
            )

        def preprocess(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            return {
                "image": images,
                "caption": examples[caption_column],
            }

        self._hf_dataset = hf_dataset["train"].with_transform(preprocess)

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
            dict: A dataset example with 2 keys: "image", and "caption".
                The "image" key maps to a `PIL` image in RGB format.
                The "caption" key maps to a string.
        """
        return self._hf_dataset[idx]

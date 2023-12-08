import typing

import datasets
import torch.utils.data


class HFHubImageCaptionDataset(torch.utils.data.Dataset):
    """An image-caption dataset for Text-to-Image datasets from the HuggingFace Hub."""

    def __init__(
        self,
        dataset_name: str,
        hf_load_dataset_kwargs: typing.Optional[dict[str, typing.Any]] = None,
        image_column: str = "image",
        caption_column: str = "text",
    ):
        """Initialize a HFHubImageCaptionDataset.

        Args:
            dataset_name (str): The HF Hub dataset name (a.k.a. path).
            hf_load_dataset_kwargs (dict[str, typing.Any], optional): kwargs to forward to `datasets.load_dataset(...)`.
            image_column (str, optional): The name of the image column in the dataset. Defaults to "image".
            caption_column (str, optional): The name of the caption column in the dataset. Defaults to "text".

        Raises:
            ValueError: If `image_column` is not found in the dataset.
            ValueError: If `caption_column` is not found in the dataset.
        """
        super().__init__()
        hf_load_dataset_kwargs = hf_load_dataset_kwargs or {}
        hf_dataset = datasets.load_dataset(dataset_name, **hf_load_dataset_kwargs)

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
            dict: A dataset example with 3 keys: "image", "caption", and "id".
                The "image" key maps to a `PIL` image in RGB format.
                The "caption" key maps to a string.
                The "id" key is the example's index (often used for caching).
        """
        example = self._hf_dataset[idx]
        example["id"] = idx
        return example

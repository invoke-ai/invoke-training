import io
import typing

import datasets
import torch.utils.data
from PIL import Image


class HFImagePairPreferenceDataset(torch.utils.data.Dataset):
    """A wrapper for the Hugging Face hub "yuvalkirstain/pickapic_v2" dataset
    (https://huggingface.co/datasets/yuvalkirstain/pickapic_v2).

    Designed to be expanded in the future to other HF image pair preference datasets.
    """

    def __init__(
        self,
        hf_dataset,
        skip_no_preference=True,
        split: str = "train",
        image_0_column: str = "jpg_0",
        label_0_column: str = "label_0",
        image_1_column: str = "jpg_1",
        label_1_column: str = "jpg_1",
        caption_column: str = "caption",
    ):
        """
        Args:
            skip_no_preference (bool, optional): If True, skip image pairs without a preference.
        """
        column_names = hf_dataset[split].column_names

        for col_name in [image_0_column, label_0_column, image_1_column, label_1_column, caption_column]:
            if col_name not in column_names:
                raise ValueError(f"Column '{col_name}' is not in the set of dataset column names: '{column_names}'.")

        eps = 0.0001

        if skip_no_preference:
            # Filter to only include pairs with a clear preference.
            def filter(example: dict[str, typing.Any]) -> bool:
                return abs(example["label_0"] - example["label_1"]) > eps

            hf_dataset = hf_dataset.filter(filter)

        def preprocess(examples):
            image_0_list = [Image.open(io.BytesIO(image)).convert("RGB") for image in examples[image_0_column]]
            image_1_list = [Image.open(io.BytesIO(image)).convert("RGB") for image in examples[image_1_column]]

            image_0_is_better = []
            image_1_is_better = []
            for label_0, label_1 in zip(examples["label_0"], examples["label_1"]):
                if (label_0 - label_1) > eps:
                    # Label 0 is better.
                    image_0_is_better.append(True)
                    image_1_is_better.append(False)
                elif (label_1 - label_0) > eps:
                    # Label 1 is better.
                    image_0_is_better.append(False)
                    image_1_is_better.append(True)
                else:
                    # Tie.
                    image_0_is_better.append(False)
                    image_1_is_better.append(False)

            return {
                "image_0": image_0_list,
                "image_1": image_1_list,
                "prefer_0": image_0_is_better,
                "prefer_1": image_1_is_better,
                "caption": examples[caption_column],
            }

        self._hf_dataset = hf_dataset[split].with_transform(preprocess)

    @classmethod
    def from_hub(
        cls,
        dataset_name: str,
        skip_no_preference: bool = True,
        split: str = "train",
        hf_load_dataset_kwargs: typing.Optional[dict[str, typing.Any]] = None,
    ):
        """Initialize a HFImageCaptionDataset from a Hugging Face Hub dataset.

        Args:
            dataset_name (str): The HF Hub dataset name (a.k.a. path).
            hf_load_dataset_kwargs (dict[str, typing.Any], optional): kwargs to forward to `datasets.load_dataset(...)`.
        """
        if dataset_name != "yuvalkirstain/pickapic_v2":
            raise NotImplementedError(
                "The HFImagePairPreferenceDataset class likely won't work with datasets other than "
                "'yuvalkirstain/pickapic_v2'."
            )

        hf_load_dataset_kwargs = hf_load_dataset_kwargs or {}
        hf_dataset = datasets.load_dataset(dataset_name, **hf_load_dataset_kwargs)

        return cls(hf_dataset=hf_dataset, skip_no_preference=skip_no_preference, split=split)

    def __len__(self) -> int:
        """Get the dataset length.

        Returns:
            int: The number of image pairs in the dataset.
        """
        return len(self._hf_dataset)

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        """Load the dataset example at index `idx`.

        Raises:
            IndexError: If `idx` is out of range.

        Returns:
            dict: A dataset example with the following keys: ["id", "image_1", "caption_1", "image_2", "caption_2",
                "prefer_1", "prefer_2"]
                The image keys map to a `PIL` image in RGB format.
                The caption keys map to strings.
                The "id" key is the example's index (often used for caching).
        """
        example = self._hf_dataset[idx]
        example["id"] = idx
        return example

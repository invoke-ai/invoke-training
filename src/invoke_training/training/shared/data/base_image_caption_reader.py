import typing


class BaseImageCaptionReader:
    """An interface for all image-caption dataset readers to implement."""

    def __len__(self) -> int:
        """Get the dataset length.

        Returns:
            int: The number of image-caption pairs in the dataset.
        """
        raise NotImplementedError("__len__() is not implemented.")

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        """Load the dataset example at index `idx`.

        Raises:
            IndexError: If `idx` is out of range.

        Returns:
            dict: A dataset example with 2 keys: "image", and "caption".
                The "image" key maps to a `PIL` image in RGB format.
                The "caption" key maps to a string.
        """
        raise NotImplementedError("__getitem__() is not implemented.")

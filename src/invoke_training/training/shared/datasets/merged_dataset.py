import typing


class MergedDataset:
    """A wrapper that merges the examples from multiple other datasets."""

    def __init__(self, datasets: list):
        """Initialize MergedDataset.

        Args:
            datasets (list): List of datasets.

        Raises:
            ValueError: If the dataset lengths do not match.
        """
        self._datasets = datasets

        # Verify that all datasets have the same length.
        dataset_lengths = [len(d) for d in self._datasets]
        if not all([length == dataset_lengths[0] for length in dataset_lengths]):
            raise ValueError(f"All dataset lengths much match: {dataset_lengths}")

    def __len__(self):
        return len(self._datasets[0])

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        example = {}
        for dataset in self._datasets:
            example.update(dataset[idx])
        return example

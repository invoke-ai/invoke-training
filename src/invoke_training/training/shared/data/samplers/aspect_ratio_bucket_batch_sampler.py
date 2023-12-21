import copy
import math
import random
from typing import Iterator

from torch.utils.data import Sampler

from invoke_training.training.shared.data.utils.aspect_ratio_bucket_manager import AspectRatioBucketManager
from invoke_training.training.shared.data.utils.resolution import Resolution

AspectRatioBuckets = dict[Resolution, list[int]]


class AspectRatioBucketBatchSampler(Sampler[list[int]]):
    """A batch sampler that adheres to aspect ratio buckets."""

    def __init__(
        self,
        buckets: AspectRatioBuckets,
        batch_size: int,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> None:
        """Initialize AspectRatioBucketBatchSampler.

        For most use cases, initialize via AspectRatioBucketBatchSampler.from_image_sizes(...).
        """
        self._buckets = buckets
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._random = random.Random(seed)

    @classmethod
    def from_image_sizes(
        cls,
        bucket_manager: AspectRatioBucketManager,
        image_sizes: list[Resolution],
        batch_size: int,
        shuffle: bool = False,
        seed: int | None = None,
    ):
        """Initialize from an AspectRatioBucketManager and the list of dataset image resolutions."""
        buckets = cls._build_bucket_to_index_map(bucket_manager, image_sizes)
        return cls(buckets=buckets, batch_size=batch_size, shuffle=shuffle, seed=seed)

    @classmethod
    def _build_bucket_to_index_map(
        cls,
        bucket_manager: AspectRatioBucketManager,
        image_sizes: list[Resolution],
    ) -> AspectRatioBuckets:
        bucket_to_indexes: AspectRatioBuckets = dict()

        for bucket_resolution in bucket_manager.buckets:
            bucket_to_indexes[bucket_resolution] = []

        for index, image_size in enumerate(image_sizes):
            aspect_ratio_bucket = bucket_manager.get_aspect_ratio_bucket(image_size)
            bucket_to_indexes[aspect_ratio_bucket].append(index)

        return bucket_to_indexes

    def get_buckets(self) -> AspectRatioBuckets:
        return copy.deepcopy(self._buckets)

    def __iter__(self) -> Iterator[list[int]]:
        batches: list[list[int]] = []

        # TODO(ryand): If self._shuffle == False, should we still shuffle just with a fixed seed every time? If we
        # don't shuffle at all then all of the batches from a bucket will be grouped together. If there's a correlation
        # between aspect ratio and image content in a dataset, this could result in unevenly distributed image content
        # over the dataset.

        for bucket_resolution in sorted(list(self._buckets.keys())):
            ordered_bucket_images = self._buckets[bucket_resolution].copy()
            if self._shuffle:
                # Shuffle the images within a bucket.
                self._random.shuffle(ordered_bucket_images)

            # Prepare batches for a single bucket.
            batch_start = 0
            while batch_start < len(ordered_bucket_images):
                batch_end = min(batch_start + self._batch_size, len(ordered_bucket_images))
                batches.append(ordered_bucket_images[batch_start:batch_end])
                batch_start += self._batch_size

        if self._shuffle:
            # We've already shuffled the images within each bucket, now we shuffle the batches.
            self._random.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        num_batches = 0
        for bucket_images in self._buckets.values():
            num_batches += math.ceil(len(bucket_images) / self._batch_size)
        return num_batches

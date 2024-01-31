from invoke_training._shared.data.samplers.aspect_ratio_bucket_batch_sampler import (
    AspectRatioBucketBatchSampler,
)
from invoke_training._shared.data.utils.aspect_ratio_bucket_manager import AspectRatioBucketManager
from invoke_training._shared.data.utils.resolution import Resolution


def assert_shuffled_samples_match(samples_1, samples_2):
    """Utility function to assert that two batch sampler outputs are equivalent aside from having been shuffled."""
    # Same number of batches.
    assert len(samples_1) == len(samples_2)
    # Same total number of examples.
    assert sum([len(b) for b in samples_1]) == sum([len(b) for b in samples_2])
    # Same set of examples.
    assert {x for batch in samples_1 for x in batch} == {x for batch in samples_2 for x in batch}


def test_aspect_ratio_bucket_batch_sampler():
    """Basic test of AspectRatioBucketBatchSampler."""
    sampler = AspectRatioBucketBatchSampler(
        buckets={Resolution(256, 768): [1, 3, 5], Resolution(512, 512): [4], Resolution(768, 256): [0, 2]},
        batch_size=2,
        shuffle=False,
        seed=None,
    )

    assert list(sampler) == [[1, 3], [5], [4], [0, 2]]


def test_aspect_ratio_bucket_batch_sampler_len():
    """Basic test of AspectRatioBucketBatchSampler len(...) function."""
    sampler = AspectRatioBucketBatchSampler(
        buckets={Resolution(256, 768): [1, 3, 5], Resolution(512, 512): [4], Resolution(768, 256): [0, 2]},
        batch_size=2,
        shuffle=False,
        seed=None,
    )

    assert len(sampler) == len(list(sampler))


def test_aspect_ratio_bucket_batch_sampler_from_image_sizes():
    """Test AspectRatioBucketBatchSampler when initialized with AspectRatioBucketBatchSampler.from_image_size(...)."""
    # Configure bucket_manager to have the following aspect ratio buckets:
    # (256, 1024), (256, 768), (512, 512), (768, 256), (1024, 768)
    bucket_manager = AspectRatioBucketManager.from_constraints(
        target_resolution=512, start_dim=256, end_dim=768, divisible_by=256
    )

    image_sizes = [
        Resolution(256, 768),  # Bucket 1 (256, 768)
        Resolution(512, 512),  # Bucket 2 (512, 512)
        Resolution(768, 256),  # Bucket 3 (768, 256)
        Resolution(264, 768),  # Bucket 1 (256, 768)
        Resolution(272, 768),  # Bucket 1 (256, 768)
        Resolution(768, 264),  # Bucket 3 (768, 256)
    ]

    sampler = AspectRatioBucketBatchSampler.from_image_sizes(
        bucket_manager=bucket_manager, image_sizes=image_sizes, batch_size=2, shuffle=False
    )

    assert list(sampler) == [[0, 3], [4], [1], [2, 5]]


def test_aspect_ratio_bucket_batch_sampler_shuffle():
    """Test AspectRatioBucketBatchSampler shuffle behavior."""
    buckets = {Resolution(256, 512): [1, 3, 5, 6, 7], Resolution(512, 512): [4], Resolution(512, 256): [0, 2]}
    batch_size = 2
    unshuffled_sampler = AspectRatioBucketBatchSampler(buckets=buckets, batch_size=batch_size, shuffle=False, seed=None)
    shuffled_sampler = AspectRatioBucketBatchSampler(buckets=buckets, batch_size=batch_size, shuffle=True, seed=None)

    unshuffled_samples = list(unshuffled_sampler)
    shuffled_samples = list(shuffled_sampler)

    assert_shuffled_samples_match(shuffled_samples, unshuffled_samples)
    # Not equal, because one is shuffled.
    assert shuffled_samples != unshuffled_samples


def test_aspect_ratio_bucket_batch_sampler_seed():
    """Test AspectRatioBucketBatchSampler seed behavior."""
    buckets = {Resolution(256, 512): [1, 3, 5, 6, 7], Resolution(512, 512): [4], Resolution(512, 256): [0, 2]}
    batch_size = 2
    base_sampler = AspectRatioBucketBatchSampler(buckets=buckets, batch_size=batch_size, shuffle=True, seed=1)
    same_seed_sampler = AspectRatioBucketBatchSampler(buckets=buckets, batch_size=batch_size, shuffle=True, seed=1)
    diff_seed_sampler = AspectRatioBucketBatchSampler(buckets=buckets, batch_size=batch_size, shuffle=True, seed=2)

    base_samples = list(base_sampler)
    same_seed_samples = list(same_seed_sampler)
    diff_seed_samples = list(diff_seed_sampler)

    # Samples generated with the same seed should match exactly.
    assert base_samples == same_seed_samples

    # Samples generated with different seeds should match, except for the example ordering.
    assert_shuffled_samples_match(base_samples, diff_seed_samples)
    assert base_samples != diff_seed_samples

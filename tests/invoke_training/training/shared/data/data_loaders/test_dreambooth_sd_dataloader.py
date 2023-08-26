import torch

from invoke_training.training.shared.data.data_loaders.dreambooth_sd_dataloader import (
    InterleavedSampler,
    SequentialRangeSampler,
    ShuffledRangeSampler,
)

#########################
# SequentialRangeSampler
#########################


def test_sequential_range_sampler():
    """Test that the SequentialRangeSampler yields the correct sequence of values."""
    sampler = SequentialRangeSampler(5, 10)

    assert list(sampler) == [5, 6, 7, 8, 9]
    # Assert that it can be iterated multiple times.
    assert list(sampler) == [5, 6, 7, 8, 9]


def test_sequential_range_sampler_len():
    """Test the SequentialRangeSampler len() function."""
    sampler = SequentialRangeSampler(5, 10)
    assert len(sampler) == 5


#######################
# ShuffledRangeSampler
#######################


def test_shuffled_range_sampler():
    """Test that the ShuffledRangeSampler yields the correct set of values, and that they are shuffled."""
    start = 5
    end = 20
    sampler = ShuffledRangeSampler(start, end)
    samples = list(sampler)

    expected = list(range(start, end))
    assert samples != expected  # The order should not match.
    assert sorted(samples) == expected  # The contents should match.


def test_shuffled_range_sampler_len():
    """Test the ShuffledRangeSampler len() function."""
    start = 5
    end = 20
    sampler = ShuffledRangeSampler(start, end)
    samples = list(sampler)

    expected = list(range(start, end))
    assert samples != expected  # The order should not match.
    assert sorted(samples) == expected  # The contents should match.


def test_shuffled_range_sampler_reshuffles():
    """Test that the ShuffledRangeSampler reshuffles after each pass over the data."""
    start = 5
    end = 20
    sampler = ShuffledRangeSampler(start, end)
    samples_1 = list(sampler)
    samples_2 = list(sampler)

    assert samples_1 != samples_2
    assert sorted(samples_1) == sorted(samples_2)


def test_shuffled_range_sampler_without_generator_is_random():
    """Test that the ShuffledRangeSampler is random when no generator is provided."""
    sampler_1 = ShuffledRangeSampler(5, 20)
    sampler_2 = ShuffledRangeSampler(5, 20)

    samples_1 = list(sampler_1)
    samples_2 = list(sampler_2)

    assert samples_1 != samples_2


def test_shuffled_range_sampler_with_generator_is_deterministic():
    """Test that the ShuffledRangeSampler is deterministic when a generator is provided."""
    # Create 2 generators with the same seed.
    generator_1 = torch.Generator()
    generator_1.manual_seed(123)
    generator_2 = torch.Generator()
    generator_2.manual_seed(123)

    sampler_1 = ShuffledRangeSampler(5, 20, generator_1)
    sampler_2 = ShuffledRangeSampler(5, 20, generator_2)

    samples_1 = list(sampler_1)
    samples_2 = list(sampler_2)

    assert samples_1 == samples_2


#####################
# InterleavedSampler
#####################


def test_interleaved_sampler():
    """Test that the InterleavedSampler yields the correct sequence."""
    sampler_1 = [0, 1, 2, 3]
    sampler_2 = [4, 5, 6]
    sampler_3 = [7, 8, 9, 10, 11, 12]

    sampler = InterleavedSampler([sampler_1, sampler_2, sampler_3])
    samples = list(sampler)

    assert samples == [0, 4, 7, 1, 5, 8, 2, 6, 9]


def test_interleaved_sampler_len():
    """Test the InterleavedSampler len() function."""
    sampler_1 = [0, 1, 2, 3]
    sampler_2 = [4, 5]
    sampler_3 = [7, 8, 9, 10, 11, 12]

    sampler = InterleavedSampler([sampler_1, sampler_2, sampler_3])
    assert len(sampler) == 2 * 3

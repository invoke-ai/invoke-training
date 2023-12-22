from invoke_training.training.shared.data.samplers.interleaved_sampler import InterleavedSampler


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

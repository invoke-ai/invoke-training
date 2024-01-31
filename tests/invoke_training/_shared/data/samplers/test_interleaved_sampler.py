from invoke_training._shared.data.samplers.interleaved_sampler import InterleavedSampler


def test_interleaved_sampler():
    """Test that the InterleavedSampler yields the correct sequence."""
    sampler_1 = [0, 1, 2, 3]
    sampler_2 = [4, 5, 6]
    sampler_3 = [7, 8, 9, 10, 11, 12]

    sampler = InterleavedSampler([sampler_1, sampler_2, sampler_3])
    samples = list(sampler)

    assert samples == [0, 4, 7, 1, 5, 8, 2, 6, 9]


def test_interleaved_sampler_batches():
    """Test that the InterleavedSampler yields the correct sequence with batch samplers."""
    sampler_1 = [[0, 1, 2], [3, 4, 5], [6]]
    sampler_2 = [[7, 8], [9]]
    sampler_3 = [[10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21]]

    sampler = InterleavedSampler([sampler_1, sampler_2, sampler_3])
    samples = list(sampler)

    assert samples == [[0, 1, 2], [7, 8], [10, 11, 12], [3, 4, 5], [9], [13, 14, 15]]


def test_interleaved_sampler_len():
    """Test the InterleavedSampler len() function."""
    sampler_1 = [0, 1, 2, 3]
    sampler_2 = [4, 5]
    sampler_3 = [7, 8, 9, 10, 11, 12]

    sampler = InterleavedSampler([sampler_1, sampler_2, sampler_3])
    assert len(sampler) == 2 * 3

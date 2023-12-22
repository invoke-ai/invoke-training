from invoke_training.training.shared.data.samplers.concat_sampler import ConcatSampler


def test_concat_sampler():
    """Test that the ConcatSampler yields the correct sequence."""
    sampler_1 = [0, 1, 2, 3]
    sampler_2 = [4, 5, 6]
    sampler_3 = [7, 8, 9, 10, 11, 12]

    sampler = ConcatSampler([sampler_1, sampler_2, sampler_3])
    samples = list(sampler)

    assert samples == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def test_concat_sampler_batches():
    """Test that the ConcatSampler yields the correct sequence with batch samplers."""
    sampler_1 = [[0, 1, 2], [3, 4, 5], [6]]
    sampler_2 = [[7, 8], [9]]
    sampler_3 = [[10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21]]

    sampler = ConcatSampler([sampler_1, sampler_2, sampler_3])
    samples = list(sampler)

    assert samples == [[0, 1, 2], [3, 4, 5], [6], [7, 8], [9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21]]


def test_concat_sampler_len():
    """Test the ConcatSampler len() function."""
    sampler_1 = [0, 1, 2, 3]
    sampler_2 = [4, 5, 6]
    sampler_3 = [7, 8, 9, 10, 11, 12]

    sampler = ConcatSampler([sampler_1, sampler_2, sampler_3])
    assert len(sampler) == 13

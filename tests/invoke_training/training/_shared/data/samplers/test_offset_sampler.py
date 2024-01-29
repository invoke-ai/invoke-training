from torch.utils.data.sampler import SequentialSampler

from invoke_training._shared.data.samplers.offset_sampler import OffsetSampler


def test_offset_sampler():
    """Test that the OffsetSampler yields the correct sequence of values."""
    sequential_sampler = SequentialSampler([0] * 5)
    offset_sampler = OffsetSampler(sampler=sequential_sampler, offset=10)

    assert list(offset_sampler) == list(range(10, 15))
    # Assert that it can be iterated multiple times.
    assert list(offset_sampler) == list(range(10, 15))


def test_offset_sampler_len():
    """Test the OffsetSampler len() function."""
    sequential_sampler = SequentialSampler([0] * 5)
    offset_sampler = OffsetSampler(sampler=sequential_sampler, offset=10)
    assert len(offset_sampler) == 5

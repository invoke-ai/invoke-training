from torch.utils.data.sampler import BatchSampler, SequentialSampler

from invoke_training._shared.data.samplers.batch_offset_sampler import BatchOffsetSampler


def test_batch_offset_sampler():
    """Test that the BatchOffsetSampler yields the correct sequence of values."""
    sequential_sampler = SequentialSampler([0] * 5)
    batch_sampler = BatchSampler(sequential_sampler, batch_size=2, drop_last=False)

    batch_offset_sampler = BatchOffsetSampler(sampler=batch_sampler, offset=10)

    assert list(batch_offset_sampler) == [[10, 11], [12, 13], [14]]
    # Assert that it can be iterated multiple times.
    assert list(batch_offset_sampler) == [[10, 11], [12, 13], [14]]


def test_batch_offset_sampler_len():
    """Test the BatchOffsetSampler len() function."""
    sequential_sampler = SequentialSampler([0] * 5)
    batch_sampler = BatchSampler(sequential_sampler, batch_size=2, drop_last=False)
    batch_offset_sampler = BatchOffsetSampler(sampler=batch_sampler, offset=10)
    assert len(batch_offset_sampler) == 3

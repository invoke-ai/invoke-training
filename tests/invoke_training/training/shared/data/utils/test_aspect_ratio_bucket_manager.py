from contextlib import nullcontext

import pytest

from invoke_training.training.shared.data.utils.aspect_ratio_bucket_manager import AspectRatioBucketManager


def test_aspect_ratio_bucket_manager():
    assert False


@pytest.mark.parametrize(
    ["target_resolution", "start_dim", "end_dim", "divisible_by", "should_raise"],
    [
        ((1024, 1024), 512, 2048, 64, False),
        ((1025, 1024), 512, 2048, 64, True),  # target_resolution[0] not divisible by divisible_by.
        ((1024, 1025), 512, 2048, 64, True),  # target_resolution[1] not divisible by divisible_by.
        ((1024, 1024), 513, 2048, 64, True),  # start_dim not divisible by divisible_by.
        ((1024, 1024), 512, 2049, 64, True),  # end_dim not divisible by divisible_by.
        ((1024, 1024), 1024, 512, 64, True),  # start_dim > end_dim.
    ],
)
def test_build_aspect_ratio_buckets_input_validation(
    target_resolution: tuple[int, int], start_dim: int, end_dim: int, divisible_by: int, should_raise: bool
):
    """Test validation of all input params to AspectRatioBucketManager.build_aspect_ratio_buckets(...)."""
    expectation = pytest.raises(AssertionError) if should_raise else nullcontext()
    with expectation:
        _ = AspectRatioBucketManager.build_aspect_ratio_buckets(
            target_resolution=target_resolution,
            start_dim=start_dim,
            end_dim=end_dim,
            divisible_by=divisible_by,
        )


@pytest.mark.parametrize(
    ["target_resolution", "start_dim", "end_dim", "divisible_by", "expected"],
    [
        # 1 bucket
        ((1024, 1024), 1024, 1024, 64, {(1024, 1024)}),
        # Multiple buckets.
        ((1024, 1024), 768, 1280, 128, {(768, 1280), (896, 1152), (1024, 1024), (1152, 896), (1280, 768)}),
    ],
)
def test_build_aspect_ratio_buckets(
    target_resolution: tuple[int, int],
    start_dim: int,
    end_dim: int,
    divisible_by: int,
    expected: set[tuple[int, int]],
):
    buckets = AspectRatioBucketManager.build_aspect_ratio_buckets(
        target_resolution=target_resolution,
        start_dim=start_dim,
        end_dim=end_dim,
        divisible_by=divisible_by,
    )

    assert buckets == expected

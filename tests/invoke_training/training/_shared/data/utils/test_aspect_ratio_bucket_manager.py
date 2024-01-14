from contextlib import nullcontext

import pytest

from invoke_training.training._shared.data.utils.aspect_ratio_bucket_manager import AspectRatioBucketManager
from invoke_training.training._shared.data.utils.resolution import Resolution


@pytest.mark.parametrize(
    ["target_resolution", "start_dim", "end_dim", "divisible_by", "should_raise"],
    [
        (1024, 512, 2048, 64, False),
        (1025, 512, 2048, 64, True),  # target_resolution not divisible by divisible_by.
        (1024, 513, 2048, 64, True),  # start_dim not divisible by divisible_by.
        (1024, 512, 2049, 64, True),  # end_dim not divisible by divisible_by.
        (1024, 1024, 512, 64, True),  # start_dim > end_dim.
    ],
)
def test_build_aspect_ratio_buckets_input_validation(
    target_resolution: int, start_dim: int, end_dim: int, divisible_by: int, should_raise: bool
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
        (1024, 1024, 1024, 64, {Resolution(1024, 1024)}),
        # Multiple buckets.
        (
            1024,
            768,
            1280,
            128,
            {
                Resolution(768, 1280),
                Resolution(896, 1152),
                Resolution(1024, 1024),
                Resolution(1152, 896),
                Resolution(1280, 768),
            },
        ),
    ],
)
def test_build_aspect_ratio_buckets(
    target_resolution: int,
    start_dim: int,
    end_dim: int,
    divisible_by: int,
    expected: set[Resolution],
):
    buckets = AspectRatioBucketManager.build_aspect_ratio_buckets(
        target_resolution=target_resolution,
        start_dim=start_dim,
        end_dim=end_dim,
        divisible_by=divisible_by,
    )

    assert buckets == expected


@pytest.mark.parametrize(
    ["resolution", "expected_bucket"],
    [
        (Resolution(1024, 1024), Resolution(1024, 1024)),  # Exact match.
        (Resolution(128, 1024), Resolution(768, 1280)),  # Small aspect ratio.
        (Resolution(1024, 128), Resolution(1280, 768)),  # Large aspect ratio.
    ],
)
def test_get_aspect_ratio_bucket(resolution: Resolution, expected_bucket: Resolution):
    arbm = AspectRatioBucketManager.from_constraints(
        target_resolution=1024, start_dim=768, end_dim=1280, divisible_by=128
    )

    nearest_bucket = arbm.get_aspect_ratio_bucket(resolution)

    assert nearest_bucket == expected_bucket

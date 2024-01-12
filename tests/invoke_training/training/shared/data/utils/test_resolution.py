import pytest

from invoke_training.training._shared.data.utils.resolution import Resolution


@pytest.mark.parametrize(
    ["input", "expected_resolution"],
    [
        (5, Resolution(5, 5)),  # From int.
        ((5, 6), Resolution(5, 6)),  # From tuple[int, int].
        (Resolution(5, 6), Resolution(5, 6)),  # From Resolution.
    ],
)
def test_resolution_parse(input, expected_resolution: Resolution):
    resolution = Resolution.parse(input)
    assert resolution == expected_resolution

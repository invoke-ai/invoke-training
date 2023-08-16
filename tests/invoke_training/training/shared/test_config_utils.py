import pytest

from invoke_training.training.shared.config_utils import flatten_config


@pytest.mark.parametrize(
    ["in_val", "expected"],
    [
        (1, {"": 1}),  # Non-iterable value
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}),  # Flat dict
        ([1, 2], {"0": 1, "1": 2}),  # Flat list
        ({"a": 1, "b": {"c": 2, "d": 3}}, {"a": 1, "b.c": 2, "b.d": 3}),  # Nested dict
        ({"a": 1, "b": [2, 3]}, {"a": 1, "b.0": 2, "b.1": 3}),  # Nested list
        ({"a": 1, "b": {"c": 2}, "d": [3, 4]}, {"a": 1, "b.c": 2, "d.0": 3, "d.1": 4}),  # Mix
    ],
)
def test_flatten_config(in_val, expected):
    assert flatten_config(in_val) == expected

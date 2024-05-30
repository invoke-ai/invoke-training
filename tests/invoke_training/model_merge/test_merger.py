import math

import pytest
import torch

from invoke_training.model_merge.merger import WeightedMergeOperation, WeightedModelMerger

from .utils import state_dicts_are_close


def test_weighted_sum_merger_raises_on_not_enough_state_dicts():
    with pytest.raises(ValueError, match="Must provide >=2 models to merge."):
        _ = WeightedModelMerger().merge([{}], [0.5])


def test_weighted_sum_merger_raises_on_mismatched_weights():
    with pytest.raises(ValueError, match="Must provide a weight for each model."):
        _ = WeightedModelMerger().merge([{}, {}], [0.5, 0.5, 0.5])


@pytest.mark.parametrize(
    ["state_dicts", "weights", "expected_state_dict", "operation"],
    [
        # Lerp.
        (
            [
                {"a": torch.tensor(1.0), "b": torch.tensor(2.0)},
                {"a": torch.tensor(3.0), "b": torch.tensor(4.0)},
            ],
            [1.0, 1.0],
            {"a": torch.tensor(2.0), "b": torch.tensor(3.0)},
            WeightedMergeOperation.Lerp,
        ),
        # Lerp with unbalanced weights.
        (
            [
                {"a": torch.tensor(1.0), "b": torch.tensor(2.0)},
                {"a": torch.tensor(3.0), "b": torch.tensor(4.0)},
            ],
            [1.0, 3.0],
            {"a": torch.tensor(1.0 * 0.25 + 3.0 * 0.75), "b": torch.tensor(2.0 * 0.25 + 4.0 * 0.75)},
            WeightedMergeOperation.Lerp,
        ),
        # Lerp with more than 2 state dicts.
        (
            [
                {"a": torch.tensor(1.0), "b": torch.tensor(2.0)},
                {"a": torch.tensor(2.0), "b": torch.tensor(3.0)},
                {"a": torch.tensor(3.0), "b": torch.tensor(4.0)},
            ],
            [1.0, 1.0, 1.0],
            {"a": torch.tensor(2.0), "b": torch.tensor(3.0)},
            WeightedMergeOperation.Lerp,
        ),
        # Slerp with scalar tensors falls back to lerp.
        (
            [
                {"a": torch.tensor(1.0), "b": torch.tensor(2.0)},
                {"a": torch.tensor(3.0), "b": torch.tensor(4.0)},
            ],
            [1.0, 1.0],
            {"a": torch.tensor(2.0), "b": torch.tensor(3.0)},
            WeightedMergeOperation.Slerp,
        ),
        # Slerp with colinear vector tensors falls back to lerp.
        (
            [
                {"a": torch.tensor([1.0, 2.0])},
                {"a": torch.tensor([2.0, 4.0])},
            ],
            [1.0, 1.0],
            {"a": torch.tensor([1.5, 3.0])},
            WeightedMergeOperation.Slerp,
        ),
        # Slerp with orthogonal vector tensors.
        (
            [
                {"a": torch.tensor([1.0, 0.0])},
                {"a": torch.tensor([0.0, 1.0])},
            ],
            [1.0, 1.0],
            {"a": torch.tensor([math.sin(math.pi / 4), math.sin(math.pi / 4)])},
            WeightedMergeOperation.Slerp,
        ),
    ],
)
def test_weighted_merger(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float],
    expected_state_dict: dict[str, torch.Tensor],
    operation: WeightedMergeOperation,
):
    merged_state_dict = WeightedModelMerger().merge(state_dicts, weights, operation=operation)
    assert state_dicts_are_close(merged_state_dict, expected_state_dict)

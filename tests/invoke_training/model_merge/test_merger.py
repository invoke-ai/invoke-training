import pytest
import torch

from invoke_training.model_merge.merger import WeightedMergeOperation, WeightedModelMerger


def state_dicts_are_close(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    """Helper function for comparing two state dicts."""
    return all(torch.allclose(a[key], b[key]) for key in a.keys())


def test_weighted_sum_merger_raises_on_not_enough_state_dicts():
    with pytest.raises(ValueError, match="Must provide >=2 models to merge."):
        _ = WeightedModelMerger().merge([{}], [0.5])


def test_weighted_sum_merger_raises_on_mismatched_weights():
    with pytest.raises(ValueError, match="Must provide a weight for each model."):
        _ = WeightedModelMerger().merge([{}, {}], [0.5, 0.5, 0.5])


@pytest.mark.parametrize(
    ["state_dicts", "weights", "expected_state_dict"],
    [
        (
            [
                {"a": torch.tensor(1.0), "b": torch.tensor(2.0)},
                {"a": torch.tensor(3.0), "b": torch.tensor(4.0)},
            ],
            [1.0, 1.0],
            {"a": torch.tensor(2.0), "b": torch.tensor(3.0)},
        ),
        (
            [
                {"a": torch.tensor(1.0), "b": torch.tensor(2.0)},
                {"a": torch.tensor(3.0), "b": torch.tensor(4.0)},
            ],
            [1.0, 3.0],
            {"a": torch.tensor(1.0 * 0.25 + 3.0 * 0.75), "b": torch.tensor(2.0 * 0.25 + 4.0 * 0.75)},
        ),
        (
            [
                {"a": torch.tensor(1.0), "b": torch.tensor(2.0)},
                {"a": torch.tensor(2.0), "b": torch.tensor(3.0)},
                {"a": torch.tensor(3.0), "b": torch.tensor(4.0)},
            ],
            [1.0, 1.0, 1.0],
            {"a": torch.tensor(2.0), "b": torch.tensor(3.0)},
        ),
    ],
)
def test_weighted_sum_merger(
    state_dicts: list[dict[str, torch.Tensor]], weights: list[float], expected_state_dict: dict[str, torch.Tensor]
):
    merged_state_dict = WeightedModelMerger().merge(state_dicts, weights, operation=WeightedMergeOperation.WeightedSum)
    assert state_dicts_are_close(merged_state_dict, expected_state_dict)

import re

import pytest
import torch

from invoke_training.model_merge.merge_ties import merge_ties

from .utils import state_dicts_are_close


def test_merge_ties_raises_on_not_enough_state_dicts():
    with pytest.raises(ValueError, match="Must provide >=2 models to merge."):
        _ = merge_ties([{}], [0.5])


def test_merge_ties_raises_on_mismatched_weights():
    with pytest.raises(ValueError, match="Must provide a weight for each model."):
        _ = merge_ties([{}, {}], [0.5, 0.5, 0.5])


def test_merge_ties_raises_on_density_out_of_range():
    with pytest.raises(ValueError, match=re.escape("Density must be in the range [0, 1]. Value provided: '1.1'.")):
        _ = merge_ties([{}, {}], [0.5, 0.5], density=1.1)


@pytest.mark.parametrize(
    ["base_state_dict", "task_state_dicts", "task_weights", "expected_state_dict"],
    [
        # TIES.
        (
            {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])},
            [
                {"a": torch.tensor([2.0, 7.0]), "b": torch.tensor([3.0, 6.0])},
                {"a": torch.tensor([7.0, 3.0]), "b": torch.tensor([3.0, 7.0])},
            ],
            [1.0, 1.0],
            # Expected task diff state dict:
            # {"a": torch.tensor([1.0, 5.0]), "b": torch.tensor([0.0, 2.0])},
            # {"a": torch.tensor([6.0, 1.0]), "b": torch.tensor([0.0, 3.0])},
            # Expected merged diff state dict:
            # {"a": torch.tensor([6.0, 5.0]), "b": torch.tensor([0.0, 2.5])},
            # Expected final result:
            {"a": torch.tensor([7.0, 7.0]), "b": torch.tensor([3.0, 6.5])},
        ),
    ],
)
def test_merge_ties(
    base_state_dict: dict[str, torch.Tensor],
    task_state_dicts: list[dict[str, torch.Tensor]],
    task_weights: list[float],
    expected_state_dict: dict[str, torch.Tensor],
):
    merged_state_dict = merge_ties(
        base_state_dict=base_state_dict, task_state_dicts=task_state_dicts, task_weights=task_weights, density=0.5
    )
    assert state_dicts_are_close(merged_state_dict, expected_state_dict)

from typing import Literal

import pytest
import torch

from invoke_training.model_merge.merge_tasks_to_base import merge_tasks_to_base_model

from .utils import state_dicts_are_close


def test_merge_raises_on_mismatched_weights():
    with pytest.raises(ValueError, match="Must provide a weight for each model."):
        _ = merge_tasks_to_base_model({}, [{}, {}], [0.5, 0.5, 0.5])


@pytest.mark.parametrize(
    ["base_state_dict", "task_state_dicts", "task_weights", "density", "merge_method", "expected_state_dict"],
    [
        # TIES.
        (
            {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])},
            [
                {"a": torch.tensor([2.0, 7.0]), "b": torch.tensor([3.0, 6.0])},
                {"a": torch.tensor([7.0, 3.0]), "b": torch.tensor([3.0, 7.0])},
            ],
            [1.0, 1.0],
            0.5,
            "TIES",
            # Expected task diff state dict:
            # {"a": torch.tensor([1.0, 5.0]), "b": torch.tensor([0.0, 2.0])},
            # {"a": torch.tensor([6.0, 1.0]), "b": torch.tensor([0.0, 3.0])},
            # Expected merged diff state dict:
            # {"a": torch.tensor([6.0, 5.0]), "b": torch.tensor([0.0, 2.5])},
            # Expected final result:
            {"a": torch.tensor([7.0, 7.0]), "b": torch.tensor([3.0, 6.5])},
        ),
        # DARE_LINEAR.
        (
            {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])},
            [
                {"a": torch.tensor([2.0, 7.0]), "b": torch.tensor([3.0, 6.0])},
                {"a": torch.tensor([7.0, 3.0]), "b": torch.tensor([3.0, 7.0])},
            ],
            [1.0, 1.0],
            # Set density to 1.0 so that we can set an expected result without having to handle seeding the RNG.
            1.0,
            "DARE_LINEAR",
            {"a": torch.tensor([8.0, 8.0]), "b": torch.tensor([3.0, 9.0])},
        ),
        # DARE_TIES.
        (
            {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])},
            [
                {"a": torch.tensor([2.0, 7.0]), "b": torch.tensor([3.0, 6.0])},
                {"a": torch.tensor([7.0, 3.0]), "b": torch.tensor([3.0, 7.0])},
            ],
            [1.0, 1.0],
            # Set density to 1.0 so that we can set an expected result without having to handle seeding the RNG.
            1.0,
            "DARE_TIES",
            {"a": torch.tensor([4.5, 5.0]), "b": torch.tensor([3.0, 6.5])},
        ),
    ],
)
def test_merge_ties(
    base_state_dict: dict[str, torch.Tensor],
    task_state_dicts: list[dict[str, torch.Tensor]],
    task_weights: list[float],
    density: float,
    merge_method: Literal["TIES", "DARE_LINEAR", "DARE_TIES"],
    expected_state_dict: dict[str, torch.Tensor],
):
    merged_state_dict = merge_tasks_to_base_model(
        base_state_dict=base_state_dict,
        task_state_dicts=task_state_dicts,
        task_weights=task_weights,
        density=density,
        merge_method=merge_method,
    )
    assert state_dicts_are_close(merged_state_dict, expected_state_dict)

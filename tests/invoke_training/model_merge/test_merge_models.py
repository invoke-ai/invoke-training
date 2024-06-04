import math
from typing import Literal

import pytest
import torch

from invoke_training.model_merge.merge_models import merge_models

from .utils import state_dicts_are_close


def test_merge_models_raises_on_not_enough_state_dicts():
    with pytest.raises(ValueError, match="Must provide >=2 models to merge."):
        _ = merge_models(state_dicts=[{}], weights=[0.5], merge_method="LERP")


def test_merge_models_raises_on_mismatched_weights():
    with pytest.raises(ValueError, match="Must provide a weight for each model."):
        _ = merge_models(state_dicts=[{}, {}], weights=[0.5, 0.5, 0.5], merge_method="LERP")


@pytest.mark.parametrize(
    ["state_dicts", "weights", "merge_method", "expected_state_dict"],
    [
        # Lerp.
        (
            [
                {"a": torch.tensor(1.0), "b": torch.tensor(2.0)},
                {"a": torch.tensor(3.0), "b": torch.tensor(4.0)},
            ],
            [1.0, 1.0],
            "LERP",
            {"a": torch.tensor(2.0), "b": torch.tensor(3.0)},
        ),
        # Lerp with unbalanced weights.
        (
            [
                {"a": torch.tensor(1.0), "b": torch.tensor(2.0)},
                {"a": torch.tensor(3.0), "b": torch.tensor(4.0)},
            ],
            [1.0, 3.0],
            "LERP",
            {"a": torch.tensor(1.0 * 0.25 + 3.0 * 0.75), "b": torch.tensor(2.0 * 0.25 + 4.0 * 0.75)},
        ),
        # Lerp with more than 2 state dicts.
        (
            [
                {"a": torch.tensor(1.0), "b": torch.tensor(2.0)},
                {"a": torch.tensor(2.0), "b": torch.tensor(3.0)},
                {"a": torch.tensor(3.0), "b": torch.tensor(4.0)},
            ],
            [1.0, 1.0, 1.0],
            "LERP",
            {"a": torch.tensor(2.0), "b": torch.tensor(3.0)},
        ),
        # Slerp with scalar tensors falls back to lerp.
        (
            [
                {"a": torch.tensor(1.0), "b": torch.tensor(2.0)},
                {"a": torch.tensor(3.0), "b": torch.tensor(4.0)},
            ],
            [1.0, 1.0],
            "SLERP",
            {"a": torch.tensor(2.0), "b": torch.tensor(3.0)},
        ),
        # Slerp with colinear vector tensors falls back to lerp.
        (
            [
                {"a": torch.tensor([1.0, 2.0])},
                {"a": torch.tensor([2.0, 4.0])},
            ],
            [1.0, 1.0],
            "SLERP",
            {"a": torch.tensor([1.5, 3.0])},
        ),
        # Slerp with orthogonal vector tensors.
        (
            [
                {"a": torch.tensor([1.0, 0.0])},
                {"a": torch.tensor([0.0, 1.0])},
            ],
            [1.0, 1.0],
            "SLERP",
            {"a": torch.tensor([math.sin(math.pi / 4), math.sin(math.pi / 4)])},
        ),
    ],
)
def test_merge_models(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float],
    merge_method: Literal["LERP", "SLERP"],
    expected_state_dict: dict[str, torch.Tensor],
):
    merged_state_dict = merge_models(state_dicts=state_dicts, weights=weights, merge_method=merge_method)
    assert state_dicts_are_close(merged_state_dict, expected_state_dict)

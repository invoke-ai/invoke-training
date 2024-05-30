from enum import Enum
from typing import Callable

import torch

# Helper types
StateDict = dict[str, torch.Tensor]
WeightedMergeFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]


def normalize_weights(weights: list[float]) -> list[float]:
    total = sum(weights)
    return [weight / total for weight in weights]


class WeightedMergeOperation(Enum):
    WeightedSum = "weighted_sum"
    Slerp = "slerp"


def get_merge_fn(operation: WeightedMergeOperation) -> WeightedMergeFn:
    if operation == WeightedMergeOperation.WeightedSum:
        return weighted_sum
    elif operation == WeightedMergeOperation.Slerp:
        raise NotImplementedError("Slerp not implemented.")
    else:
        raise ValueError(f"Unknown operation: {operation}")


class WeightedModelMerger:
    """A class for merging 2+ models with weights."""

    def merge(self, state_dicts: list[StateDict], weights: list[float], operation: WeightedMergeOperation) -> StateDict:
        if len(state_dicts) < 2:
            raise ValueError("Must provide >=2 models to merge.")

        if len(state_dicts) != len(weights):
            raise ValueError("Must provide a weight for each model.")

        merge_fn = get_merge_fn(operation)

        normalized_weights = normalize_weights(weights)

        out_state_dict: StateDict = state_dicts[0].copy()
        out_state_dict_weight = normalized_weights[0]
        for state_dict, normalized_weight in zip(state_dicts[1:], normalized_weights[1:], strict=True):
            if state_dict.keys() != out_state_dict.keys():
                raise ValueError("State dicts must have the same keys.")

            cur_pair_weights = normalize_weights([out_state_dict_weight, normalized_weight])
            for key in out_state_dict.keys():
                out_state_dict[key] = merge_fn(out_state_dict[key], state_dict[key], cur_pair_weights[0])

            # Update the weight of out_state_dict to be the sum of all state dicts merged so far.
            out_state_dict_weight += normalized_weight

        return out_state_dict


def weighted_sum(a: torch.Tensor, b: torch.Tensor, weight_a: float) -> torch.Tensor:
    return a * weight_a + b * (1 - weight_a)

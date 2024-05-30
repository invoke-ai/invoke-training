from enum import Enum
from typing import Callable

import torch

from invoke_training.model_merge.utils.normalize_weights import normalize_weights

# Helper types
StateDict = dict[str, torch.Tensor]
WeightedMergeFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]


class WeightedMergeOperation(Enum):
    # Linear interpolation (aka weighted sum).
    Lerp = "lerp"
    # Spherical linear interpolation.
    Slerp = "slerp"


def get_merge_fn(operation: WeightedMergeOperation) -> WeightedMergeFn:
    if operation == WeightedMergeOperation.Lerp:
        return lerp
    elif operation == WeightedMergeOperation.Slerp:
        return slerp
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


def lerp(a: torch.Tensor, b: torch.Tensor, weight_a: float) -> torch.Tensor:
    """Linear interpolation."""
    return torch.lerp(a, b, (1.0 - weight_a))


def slerp(a: torch.Tensor, b: torch.Tensor, weight_a: float, dot_product_thres=0.9995, epsilon=1e-10):
    """Spherical linear interpolation."""
    # TODO(ryand): For multi-dimensional matrices, it might be better to apply slerp on a subset of the dimensions
    # (e.g. per-row), rather than treating the entire matrix as a single flattened vector.

    # Normalize the vectors.
    a_norm = torch.linalg.norm(a)
    b_norm = torch.linalg.norm(b)
    a_normalized = a / a_norm
    b_normalized = b / b_norm

    if a_norm < epsilon or b_norm < epsilon:
        # If either vector is very small, fallback to lerp to avoid weird effects.
        # TODO(ryand): Is fallback here necessary?
        return lerp(a, b, weight_a)

    # Dot product of the normalized vectors.
    # We are effectively treating multi-dimensional tensors as flattened vectors.
    dot_prod = torch.sum(a_normalized * b_normalized)

    # If the absolute value of the dot product is almost 1, the vectors are ~colinear, so use lerp.
    if torch.abs(dot_prod) > dot_product_thres:
        return lerp(a, b, weight_a)

    # Calculate initial angle between v0 and v1.
    theta_0 = torch.acos(dot_prod)

    # Angle at timestep t.
    t = 1.0 - weight_a
    theta_t = theta_0 * t

    sin_theta_0 = torch.sin(theta_0)
    sin_theta_t = torch.sin(theta_t)

    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    result = s0 * a + s1 * b

    return result

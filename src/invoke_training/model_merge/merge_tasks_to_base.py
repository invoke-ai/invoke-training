from typing import Literal

import torch
import tqdm
from peft.utils.merge_utils import dare_linear, dare_ties, ties


@torch.no_grad()
def merge_tasks_to_base_model(
    base_state_dict: dict[str, torch.Tensor],
    task_state_dicts: list[dict[str, torch.Tensor]],
    task_weights: list[float],
    density: float = 0.2,
    merge_method: Literal["TIES", "DARE_LINEAR", "DARE_TIES"] = "TIES",
) -> torch.Tensor:
    """Merge a base model with one or more task-specific models.

    Args:
        base_state_dict (dict[str, torch.Tensor]): The base state dict to merge with.
        task_state_dicts (list[dict[str, torch.Tensor]]): A list of task-specific state dicts to merge into the base
            state dict.
        task_weights (list[float]): Weights for each task state dict. Weights of 1.0 for all task_state_dicts are
            recommended as a starting point (e.g. [1.0, 1.0, 1.0]). The weights can be adjusted from there (e.g.
            [1.0, 1.3, 1.0]). The weights are multipliers applied to the diff between each task_state_dict and the base
            model.
        density (float, optional): The fraction of values to preserve in the prune/trim step of DARE/TIES methods.
            Should be in the range [0, 1].
        merge_method (Literal["TIES", "DARE_LINEAR", "DARE_TIES"], optional): The method to use for merging. Options:
            - "TIES": Use the TIES method (https://arxiv.org/pdf/2306.01708)
            - "DARE_LINEAR": Use the DARE method with linear interpolation (https://arxiv.org/pdf/2311.03099)
            - "DARE_TIES": Use the DARE method for pruning, and the TIES method for merging.
    """
    if len(task_state_dicts) != len(task_weights):
        raise ValueError("Must provide a weight for each model.")

    task_weights = torch.tensor(task_weights)

    # Choose the merging method.
    if merge_method == "TIES":
        merge_fn = ties
    elif merge_method == "DARE_LINEAR":
        merge_fn = dare_linear
    elif merge_method == "DARE_TIES":
        merge_fn = dare_ties
    else:
        raise ValueError(f"Unknown merge method: {merge_method}")

    out_state_dict: dict[str, torch.Tensor] = {}
    for key in tqdm.tqdm(base_state_dict.keys()):
        base_tensor = base_state_dict[key]
        orig_dtype = base_tensor.dtype

        # Calculate the diff between each task tensor and the base tensor.
        task_diff_tensors = [state_dict[key] - base_tensor for state_dict in task_state_dicts]

        merged_diff_tensor = merge_fn(
            task_tensors=task_diff_tensors,
            weights=task_weights,
            density=density,
        )

        # Some of the merge_fn implementations may return a tensor with a different dtype than the original tensors.
        # We cast the merged_diff_tensor back to the original dtype here.
        out_state_dict[key] = (base_tensor + merged_diff_tensor).to(dtype=orig_dtype)

    return out_state_dict

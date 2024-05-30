import torch
from peft.utils.merge_utils import ties


def merge_ties(
    base_state_dict: dict[str, torch.Tensor],
    task_state_dicts: list[dict[str, torch.Tensor]],
    task_weights: list[float],
    density: float = 0.2,
) -> torch.Tensor:
    """TIES-Merging algorithm from https://arxiv.org/pdf/2306.01708.

    Args:
        base_state_dict (dict[str, torch.Tensor]): The base state dict to merge with.
        task_state_dicts (list[dict[str, torch.Tensor]]): A list of task-specific state dicts to merge into the base
            state dict.
        task_weights (list[float]): Weights for each task state dict. Weights of 1.0 for all task_state_dicts are
            recommended as a starting point. The weights can be adjusted from there (e.g. [1.0, 1.3, 1.0]). The weights
            are multipliers applied to the diff between each task_state_dict and the base model.
        density (float, optional): The fraction of values to preserve in the "TRIM" step of TIES. Should be in the
            range [0, 1].
    """
    if len(task_state_dicts) != len(task_weights):
        raise ValueError("Must provide a weight for each model.")

    if density < 0 or density > 1:
        raise ValueError(f"Density must be in the range [0, 1]. Value provided: '{density}'.")

    task_weights = torch.tensor(task_weights)

    out_state_dict: dict[str, torch.Tensor] = {}
    for key in base_state_dict.keys():
        base_tensor = base_state_dict[key]
        # Calculate the diff between each task tensor and the base tensor.
        task_diff_tensors = [state_dict[key] - base_tensor for state_dict in task_state_dicts]

        merged_diff_tensor = ties(
            task_tensors=task_diff_tensors,
            weights=task_weights,
            density=density,
            majority_sign_method="total",
        )

        out_state_dict[key] = base_tensor + merged_diff_tensor

    return out_state_dict

import torch
from peft.utils.merge_utils import ties


def merge_ties(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float],
    density: float = 0.2,
) -> torch.Tensor:
    """TIES-Merging algorithm from https://arxiv.org/pdf/2306.01708.

    Args:
        state_dicts (list[dict[str, torch.Tensor]]): A list of state dicts to merge.
        weights (list[float]): Weights for each state dict. Weights of 1.0 for all state_dicts are recommended as a
            starting point. The relative weights can be adjusted from there (e.g. [1.0, 1.3, 1.0]). The weights are not
            normalized like in a basic Lerp merge operation, so deviating too far from a weight of 1.0 may produce
            unexpected results.
        density (float, optional): The fraction of values to preserve in the "TRIM" step of TIES. Should be in the
            range [0, 1].
    """
    if len(state_dicts) < 2:
        raise ValueError("Must provide >=2 models to merge.")

    if len(state_dicts) != len(weights):
        raise ValueError("Must provide a weight for each model.")

    if density < 0 or density > 1:
        raise ValueError(f"Density must be in the range [0, 1]. Value provided: '{density}'.")

    out_state_dict: dict[str, torch.Tensor] = {}
    for key in state_dicts[0].keys():
        # TODO(ryand): The PEFT TIES implementation uses a naive weighting strategy. It could be worth testing a
        # lerp-like weighting strategy in the "Disjoint Merge" operation.
        out_state_dict[key] = ties(
            task_tensors=[state_dict[key] for state_dict in state_dicts],
            weights=torch.tensor(weights),
            density=density,
            majority_sign_method="total",
        )

    return out_state_dict

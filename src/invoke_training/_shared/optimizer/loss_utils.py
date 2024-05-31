from typing import Literal

import torch


def reduce_loss(batch_loss: torch.Tensor, reduction_mode: Literal["mean", "sum"] = "mean") -> torch.Tensor:
    if reduction_mode == "mean":
        return batch_loss.mean()
    elif reduction_mode == "sum":
        return batch_loss.sum() / 4.0
    else:
        raise ValueError(f"Unexpected reduction_mode: '{reduction_mode}'.")

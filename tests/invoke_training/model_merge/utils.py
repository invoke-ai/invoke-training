import torch


def state_dicts_are_close(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    """Helper function for comparing two state dicts."""
    return all(torch.allclose(a[key], b[key]) for key in a.keys())

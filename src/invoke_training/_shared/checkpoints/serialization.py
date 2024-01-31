import typing
from pathlib import Path

import safetensors.torch
import torch


def save_state_dict(state_dict: typing.Dict[str, torch.Tensor], out_file: typing.Union[Path, str]):
    """Save a state_dict to a file.

    Both safetensors and torch formats are supported. The format is inferred from the `out_file` extension.
    Supported extensions:
    - ".ckpt" -> torch
    - ".pt" -> torch
    - ".safetensors -> safetensors

    Args:
        state_dict (typing.Dict[str, torch.Tensor]): The state_dict to save.
        out_file (Path | str): The output file to save to.

    Raises:
        ValueError: If the `out_file` has an unsupported file extension.
    """
    out_file = Path(out_file)
    if out_file.suffix == ".ckpt" or out_file.suffix == ".pt":
        torch.save(state_dict, out_file)
    elif out_file.suffix == ".safetensors":
        safetensors.torch.save_file(state_dict, out_file)
    else:
        raise ValueError(f"Unsupported file extension: '{out_file.suffix}'.")


def load_state_dict(in_file: typing.Union[Path, str]) -> typing.Dict[str, torch.Tensor]:
    """Load a state_dict from a file.

    Both safetensors and torch formats are supported. The format is inferred from the `in_file` extension.
    Supported extensions:
    - ".ckpt" -> torch
    - ".pt" -> torch
    - ".safetensors -> safetensors

    Args:
        in_file (Path | str): The input file to load from.

    Raises:
        ValueError: If the `in_file` has an unsupported file extension.

    Returns:
        typing.Dict[str, torch.Tensor]: The loaded state_dict.
    """
    in_file = Path(in_file)
    if in_file.suffix == ".ckpt" or in_file.suffix == ".pt":
        return torch.load(in_file)
    elif in_file.suffix == ".safetensors":
        return safetensors.torch.load_file(in_file)
    else:
        raise ValueError(f"Unsupported file extension: '{in_file.suffix}'.")

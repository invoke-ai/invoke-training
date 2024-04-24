import logging
import os

import datasets
import diffusers
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter, get_logger
from accelerate.utils import ProjectConfiguration


def initialize_accelerator(
    out_dir: str, gradient_accumulation_steps: int, mixed_precision: str, log_with: str
) -> Accelerator:
    """Configure Hugging Face accelerate and return an Accelerator.

    Args:
        out_dir (str): The output directory where results will be written.
        gradient_accumulation_steps (int): Forwarded to accelerat.Accelerator(...).
        mixed_precision (str): Forwarded to accelerate.Accelerator(...).
        log_with (str): Forwarded to accelerat.Accelerator(...)

    Returns:
        Accelerator
    """
    accelerator_project_config = ProjectConfiguration(
        project_dir=out_dir,
        logging_dir=os.path.join(out_dir, "logs"),
    )
    return Accelerator(
        project_config=accelerator_project_config,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=log_with,
    )


def initialize_logging(logger_name: str, accelerator: Accelerator) -> MultiProcessAdapter:
    """Configure logging.

    Returns an accelerate logger with multi-process logging support. Logging is configured to be more verbose on the
    main process. Non-main processes only log at error level for Hugging Face libraries (datasets, transformers,
    diffusers).

    Args:
        accelerator (Accelerator): The Accelerator to configure.

    Returns:
        MultiProcessAdapter: _description_
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # Only log errors from non-main processes.
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    return get_logger(logger_name)


def get_mixed_precision_dtype(accelerator: Accelerator):
    """Extract torch.dtype from Accelerator config.

    Args:
        accelerator (Accelerator): The Hugging Face Accelerator.

    Raises:
        NotImplementedError: If the accelerator's mixed_precision configuration is not recognized.

    Returns:
        torch.dtype: The weight type inferred from the accelerator mixed_precision configuration.
    """
    weight_dtype: torch.dtype = torch.float32
    if accelerator.mixed_precision is None or accelerator.mixed_precision == "no":
        weight_dtype = torch.float32
    elif accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        raise NotImplementedError(f"mixed_precision mode '{accelerator.mixed_precision}' is not yet supported.")
    return weight_dtype


def get_dtype_from_str(dtype_str: str) -> torch.dtype:
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    else:
        raise ValueError(f"Unrecognized dtype string: '{dtype_str}'.")

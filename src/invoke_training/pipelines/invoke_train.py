import logging

import torch

from invoke_training.config.pipeline_config import PipelineConfig
from invoke_training.pipelines._experimental.sd_dpo_lora.train import train as train_sd_ddpo_lora
from invoke_training.pipelines.callbacks import PipelineCallbacks
from invoke_training.pipelines.stable_diffusion.lora.train import train as train_sd_lora
from invoke_training.pipelines.stable_diffusion.textual_inversion.train import train as train_sd_ti
from invoke_training.pipelines.stable_diffusion_xl.lora.train import train as train_sdxl_lora
from invoke_training.pipelines.stable_diffusion_xl.lora_and_textual_inversion.train import (
    train as train_sdxl_lora_and_ti,
)
from invoke_training.pipelines.stable_diffusion_xl.textual_inversion.train import train as train_sdxl_ti


def check_cudnn(logger: logging.Logger) -> None:
    """Check for cuDNN issues that could be causing degraded performance."""
    if not torch.cuda.is_available():
        return

    if torch.backends.cudnn.is_available():
        try:
            # Note: At the time of writing (torch 2.2.1), torch.backends.cudnn.version() only raises an error the first
            # time it is called. Subsequent calls will return the version number without complaining about a mismatch.
            cudnn_version = torch.backends.cudnn.version()
            logger.info(f"cuDNN version: {cudnn_version}")
        except RuntimeError as e:
            logger.warning(
                "Encountered a cuDNN version issue. This may result in degraded performance. This issue is usually "
                "caused by an incompatible cuDNN version installed in your python environment, or on the host "
                f"system. Full error message:\n{e}"
            )
    else:
        logger.warning("cuDNN is not available. This may result in degraded performance.")


def train(config: PipelineConfig, callbacks: list[PipelineCallbacks] | None = None):
    """This is the main entry point for all training pipelines."""

    check_cudnn(logging.getLogger(__name__))

    # Fail early if invalid callback types are provided, rather than failing later when the callbacks are used.
    for cb in callbacks or []:
        assert isinstance(cb, PipelineCallbacks)

    if config.type == "SD_LORA":
        train_sd_lora(config, callbacks)
    elif config.type == "SDXL_LORA":
        train_sdxl_lora(config, callbacks)
    elif config.type == "SD_TEXTUAL_INVERSION":
        train_sd_ti(config, callbacks)
    elif config.type == "SDXL_TEXTUAL_INVERSION":
        train_sdxl_ti(config, callbacks)
    elif config.type == "SDXL_LORA_AND_TEXTUAL_INVERSION":
        train_sdxl_lora_and_ti(config, callbacks)
    elif config.type == "SD_DIRECT_PREFERENCE_OPTIMIZATION_LORA":
        print(f"Running EXPERIMENTAL pipeline: '{config.type}'.")
        train_sd_ddpo_lora(config, callbacks)
    else:
        raise ValueError(f"Unexpected pipeline type: '{config.type}'.")

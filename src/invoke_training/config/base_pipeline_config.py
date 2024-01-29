import typing
from typing import Optional

from invoke_training.config.config_base_model import ConfigBaseModel


class BasePipelineConfig(ConfigBaseModel):
    """A base config with fields that should be inherited by all pipelines."""

    type: str

    seed: Optional[int] = None
    """A randomization seed for reproducible training. Set to any constant integer for consistent training results. If
    set to `null`, training will be non-deterministic.
    """

    base_output_dir: str
    """The output directory where the training outputs (model checkpoints, logs, intermediate predictions) will be
    written. A subdirectory will be created with a timestamp for each new training run.
    """

    report_to: typing.Literal["all", "tensorboard", "wandb", "comet_ml"] = "tensorboard"
    """The integration to report results and logs to. This value is passed to Hugging Face Accelerate. See
    `accelerate.Accelerator.log_with` for more details.
    """

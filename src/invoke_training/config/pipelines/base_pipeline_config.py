from typing import Optional

from invoke_training.config.shared.config_base_model import ConfigBaseModel
from invoke_training.config.shared.training_output_config import TrainingOutputConfig


class BasePipelineConfig(ConfigBaseModel):
    """A base config with fields that should be inherited by all pipelines."""

    type: str

    seed: Optional[int] = None
    """A randomization seed for reproducible training. Set to any constant integer for consistent training results. If
    set to `null`, training will be non-deterministic.
    """

    output: TrainingOutputConfig
    """Configuration for the training run outputs (output directory, log format, checkpoint format, etc.).

    See [`TrainingOutputConfig`][invoke_training.config.shared.training_output_config.TrainingOutputConfig] for details.
    """

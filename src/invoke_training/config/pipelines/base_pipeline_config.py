from typing import Optional

from pydantic import BaseModel

from invoke_training.config.shared.training_output_config import TrainingOutputConfig


class BasePipelineConfig(BaseModel):
    """A base config with fields that should be inherited by all pipelines."""

    type: str

    seed: Optional[int] = None
    """A seed for reproducible training.
    """

    output: TrainingOutputConfig

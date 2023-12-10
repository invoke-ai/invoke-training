from typing import Optional

from pydantic import BaseModel

from invoke_training.config.shared.training_output_config import TrainingOutputConfig


class BasePipelineConfig(BaseModel):
    """A base config with fields that should be inherited by all pipelines."""

    type: str

    # A seed for reproducible training.
    seed: Optional[int] = None

    output: TrainingOutputConfig

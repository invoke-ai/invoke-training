from abc import ABC, abstractmethod

from invoke_training.config.pipeline_config import PipelineConfig


class PipelineConfigOverride(ABC):
    """An abstract class for override logic that modifies a PipelineConfig."""

    @abstractmethod
    def apply_override(self, config: PipelineConfig):
        pass

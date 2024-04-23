from invoke_training.config.pipeline_config import PipelineConfig
from invoke_training.scripts._experimental.presets.pipeline_config_override import PipelineConfigOverride


class AnyFieldOverride(PipelineConfigOverride):
    """A basic PipelineConfig override that sets any field in the PipelineConfig."""

    def __init__(self, field_name: str, value: any):
        self._field_name = field_name
        self._value = value

    def apply_override(self, config: PipelineConfig):
        field_names = self._field_name.split(".")
        try:
            for field_name in field_names[:-1]:
                config = getattr(config, field_name)
            setattr(config, field_names[-1], self._value)
        except AttributeError:
            raise ValueError(f"Field '{self._field_name}' not found in PipelineConfig.")

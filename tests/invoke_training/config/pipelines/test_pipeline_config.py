import pytest
import yaml
from pydantic import TypeAdapter

from invoke_training.config.pipeline_config import PipelineConfig

from .config_file_paths import get_pipeline_config_file_paths


@pytest.mark.parametrize("config_file", get_pipeline_config_file_paths())
def test_pipeline_config(config_file: str):
    """Test that all sample pipeline configs can be parsed as PipelineConfigs."""

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    pipeline_adapter: TypeAdapter[PipelineConfig] = TypeAdapter(PipelineConfig)

    try:
        _ = pipeline_adapter.validate_python(cfg)
    except Exception as e:
        raise Exception(f"Error parsing config file: {config_file}") from e

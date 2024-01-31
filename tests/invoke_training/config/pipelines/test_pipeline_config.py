import glob
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from invoke_training.config.pipeline_config import PipelineConfig


def test_pipeline_config():
    """Test that all sample pipeline configs can be parse as PipelineConfigs."""
    cur_file = Path(__file__)
    config_dir = cur_file.parent.parent.parent.parent.parent / "configs"
    config_files = glob.glob(str(config_dir) + "/**/*.yaml", recursive=True)

    assert len(config_files) > 0

    for config_file in config_files:
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)

        pipeline_adapter: TypeAdapter[PipelineConfig] = TypeAdapter(PipelineConfig)

        try:
            _ = pipeline_adapter.validate_python(cfg)
        except Exception as e:
            raise Exception(f"Error parsing config file: {config_file}") from e

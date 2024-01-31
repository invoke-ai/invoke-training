from pathlib import Path

import yaml
from pydantic import TypeAdapter

from invoke_training.config.pipeline_config import PipelineConfig


def get_config_dir_path() -> Path:
    p = Path(__file__).parent.parent.parent.parent / "configs"
    if not p.exists():
        raise FileNotFoundError(f"Config directory not found: '{p}'")
    return p


def load_config_from_yaml(file_path: Path | str) -> PipelineConfig:
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        cfg = yaml.safe_load(f)

    pipeline_adapter: TypeAdapter[PipelineConfig] = TypeAdapter(PipelineConfig)
    train_config = pipeline_adapter.validate_python(cfg)

    return train_config

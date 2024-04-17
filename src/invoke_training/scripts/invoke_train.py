import argparse
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from invoke_training.config.pipeline_config import PipelineConfig
from invoke_training.pipelines.invoke_train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Run a training pipeline.")
    parser.add_argument(
        "-c",
        "--cfg-file",
        type=Path,
        required=True,
        help="Path to the YAML training config file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load YAML config file.
    with open(args.cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    pipeline_adapter: TypeAdapter[PipelineConfig] = TypeAdapter(PipelineConfig)
    train_config = pipeline_adapter.validate_python(cfg)

    train(train_config)


if __name__ == "__main__":
    main()

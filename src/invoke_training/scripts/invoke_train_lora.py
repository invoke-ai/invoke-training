import argparse
from pathlib import Path

import yaml

from invoke_training.training.lora.lora_training import run_lora_training
from invoke_training.training.lora.lora_training_config import LoRATrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA model training.")
    parser.add_argument(
        "--cfg-file",
        type=Path,
        required=True,
        help="Path to the YAML training config file. See `LoRATrainingConfig` for the supported fields.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load YAML config file.
    with open(args.cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    train_config = LoRATrainingConfig(**cfg)

    run_lora_training(train_config)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import yaml

from invoke_training.config.pipelines.finetune_lora_config import FinetuneLoRAConfig
from invoke_training.training.pipelines.stable_diffusion.finetune_lora_sd import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Finetuning with LoRA for Stable Diffusion v1 and v2 base models.")
    parser.add_argument(
        "--cfg-file",
        type=Path,
        required=True,
        help="Path to the YAML training config file. See `FinetuneLoRAConfig` for the supported fields.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load YAML config file.
    with open(args.cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    train_config = FinetuneLoRAConfig(**cfg)

    run_training(train_config)


if __name__ == "__main__":
    main()

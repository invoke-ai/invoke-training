import argparse
from pathlib import Path

import yaml

from invoke_training.config.pipelines.finetune_lora_config import FinetuneLoRASDXLConfig
from invoke_training.training.finetune_lora.finetune_lora_sdxl import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Finetuning with LoRA for Stable Diffusion XL models.")
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

    train_config = FinetuneLoRASDXLConfig(**cfg)

    run_training(train_config)


if __name__ == "__main__":
    main()

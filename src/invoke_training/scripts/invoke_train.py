import argparse
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from invoke_training.config.pipeline_config import PipelineConfig
from invoke_training.pipelines._experimental.sd_dpo_lora.train import train as train_sd_ddpo_lora
from invoke_training.pipelines.stable_diffusion.lora.train import train as train_sd_lora
from invoke_training.pipelines.stable_diffusion.textual_inversion.train import train as train_sd_ti
from invoke_training.pipelines.stable_diffusion_xl.lora.train import train as train_sdxl_lora
from invoke_training.pipelines.stable_diffusion_xl.lora_and_textual_inversion.train import (
    train as train_sdxl_lora_and_ti,
)
from invoke_training.pipelines.stable_diffusion_xl.textual_inversion.train import train as train_sdxl_ti


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

    if train_config.type == "SD_LORA":
        train_sd_lora(train_config)
    elif train_config.type == "SDXL_LORA":
        train_sdxl_lora(train_config)
    elif train_config.type == "SD_TEXTUAL_INVERSION":
        train_sd_ti(train_config)
    elif train_config.type == "SDXL_TEXTUAL_INVERSION":
        train_sdxl_ti(train_config)
    elif train_config.type == "SDXL_LORA_AND_TEXTUAL_INVERSION":
        train_sdxl_lora_and_ti(train_config)
    elif train_config.type == "SD_DIRECT_PREFERENCE_OPTIMIZATION_LORA":
        print(f"Running EXPERIMENTAL pipeline: '{train_config.type}'.")
        train_sd_ddpo_lora(train_config)
    else:
        raise ValueError(f"Unexpected pipeline type: '{train_config.type}'.")


if __name__ == "__main__":
    main()

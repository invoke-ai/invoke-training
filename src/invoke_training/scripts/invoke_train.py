import argparse
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from invoke_training.config.pipelines.pipeline_config import PipelineConfig
from invoke_training.pipelines._experimental.sd_dpo_lora.train import (
    run_training as run_diffusion_dpo_sd,
)
from invoke_training.pipelines.stable_diffusion.lora.train import run_training as run_finetune_lora_sd
from invoke_training.pipelines.stable_diffusion.textual_inversion.train import (
    run_training as run_textual_inversion_sd,
)
from invoke_training.pipelines.stable_diffusion_xl.lora.train import run_training as run_finetune_lora_sdxl
from invoke_training.pipelines.stable_diffusion_xl.lora_and_textual_inversion.train import (
    run_training as run_finetune_lora_and_ti_sdxl,
)
from invoke_training.pipelines.stable_diffusion_xl.textual_inversion.train import (
    run_training as run_textual_inversion_sdxl,
)


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

    if train_config.type == "FINETUNE_LORA_SD":
        run_finetune_lora_sd(train_config)
    elif train_config.type == "FINETUNE_LORA_SDXL":
        run_finetune_lora_sdxl(train_config)
    elif train_config.type == "TEXTUAL_INVERSION_SD":
        run_textual_inversion_sd(train_config)
    elif train_config.type == "TEXTUAL_INVERSION_SDXL":
        run_textual_inversion_sdxl(train_config)
    elif train_config.type == "FINETUNE_LORA_AND_TI_SDXL":
        run_finetune_lora_and_ti_sdxl(train_config)
    elif train_config.type == "DIRECT_PREFERENCE_OPTIMIZATION_LORA_SD":
        print(f"Running EXPERIMENTAL pipeline: '{train_config.type}'.")
        run_diffusion_dpo_sd(train_config)
    else:
        raise ValueError(f"Unexpected pipeline type: '{train_config.type}'.")


if __name__ == "__main__":
    main()

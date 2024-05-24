import argparse
import logging
from pathlib import Path
from typing import Literal

import torch

from invoke_training._shared.stable_diffusion.model_loading_utils import PipelineVersionEnum, load_pipeline


# TODO(ryand): Consolidate multiple implementations of this function across the project.
def str_to_dtype(dtype_str: Literal["float32", "float16", "bfloat16"]):
    if dtype_str == "float23":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unexpected dtype: {dtype_str}")


def merge_lora_into_sd_model(
    logger: logging.Logger,
    base_model: str,
    base_model_variant: str | None,
    base_model_type: PipelineVersionEnum,
    lora_models: list[str],
    output: str,
    save_dtype: str,
):
    pipeline = load_pipeline(
        logger=logger, model_name_or_path=base_model, pipeline_version=base_model_type, variant=base_model_variant
    )
    save_dtype = str_to_dtype(save_dtype)

    logger.info(f"Loaded base model: '{base_model}'.")

    pipeline.to(save_dtype)

    lora_adapter_names = []
    for i, lora_model in enumerate(lora_models):
        lora_adapter_name = f"lora_{i}"
        pipeline.load_lora_weights(lora_model, adapter_name=lora_adapter_name)
        lora_adapter_names.append(lora_adapter_name)

    logger.info(f"Loaded {len(lora_models)} LoRA models.")

    pipeline.set_adapters(adapter_names=lora_adapter_names, adapter_weights=[1.0] * len(lora_adapter_names))
    pipeline.fuse_lora()

    output_path = Path(output)
    output_path.mkdir(parents=True)

    # TODO(ryand): Should we keep the base model variant? This is clearly a flawed assumption.
    pipeline.save_pretrained(output_path, variant=base_model_variant)
    logger.info(f"Saved merged model to '{output_path}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        type=str,
        help="The base model to merge LoRAs into. Both Hugging Face names and local paths are supported.",
        required=True,
    )
    parser.add_argument(
        "--base-model-variant",
        type=str,
        default=None,
        help="The Hugging Face Hub variant of the base model (E.g. 'fp16'). Optional.",
    )
    # TODO(ryand): Auto-detect the base-model-type.
    parser.add_argument(
        "--base-model-type",
        type=str,
        choices=["SD", "SDXL"],
        help="The type of the base model ['SD', 'SDXL'].",
    )
    parser.add_argument(
        "--lora-model",
        type=str,
        nargs="+",
        help="The path(s) to one or more LoRA models to merge into the base model.",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="The path to an output directory where the merged model will be saved (in diffusers format).",
    )
    parser.add_argument(
        "--save-dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="The dtype to save the model as.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    merge_lora_into_sd_model(
        logger=logger,
        base_model=args.base_model,
        base_model_variant=args.base_model_variant,
        base_model_type=PipelineVersionEnum(args.base_model_type),
        lora_models=args.lora_model,
        output=args.output,
        save_dtype=args.save_dtype,
    )


if __name__ == "__main__":
    main()

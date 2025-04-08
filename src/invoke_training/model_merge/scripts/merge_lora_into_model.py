import argparse  # noqa: I001
import logging
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

# fmt: off
# HACK(ryand): Import order matters, because invokeai contains circular imports.
from invokeai.backend.model_manager import BaseModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.sd_lora_conversion_utils import \
    lora_model_from_sd_state_dict
from invokeai.backend.util.original_weights_storage import \
    OriginalWeightsStorage
from safetensors.torch import load_file

# fmt: on
from invoke_training._shared.accelerator.accelerator_utils import get_dtype_from_str
from invoke_training._shared.stable_diffusion.model_loading_utils import PipelineVersionEnum, load_pipeline
from invoke_training.model_merge.utils.parse_model_arg import parse_model_arg


def to_invokeai_base_model_type(model_type: PipelineVersionEnum):
    if model_type == PipelineVersionEnum.SD:
        return BaseModelType.StableDiffusion1
    elif model_type == PipelineVersionEnum.SDXL:
        return BaseModelType.StableDiffusionXL
    else:
        raise ValueError(f"Unexpected model_type: {model_type}")


@torch.no_grad()
def merge_lora_into_sd_model(
    logger: logging.Logger,
    model_type: PipelineVersionEnum,
    base_model: str,
    base_model_variant: str | None,
    lora_models: list[tuple[str, float]],
    output: str,
    save_dtype: str,
):
    pipeline: StableDiffusionXLPipeline | StableDiffusionPipeline = load_pipeline(
        logger=logger, model_name_or_path=base_model, pipeline_version=model_type, variant=base_model_variant
    )
    save_dtype = get_dtype_from_str(save_dtype)

    logger.info(f"Loaded base model: '{base_model}'.")

    pipeline.to(save_dtype)

    models: list[torch.nn.Module] = []
    lora_prefixes: list[str] = []
    if isinstance(pipeline, StableDiffusionPipeline):
        models = [pipeline.unet, pipeline.text_encoder]
        lora_prefixes = ["lora_unet_", "lora_te_"]
    elif isinstance(pipeline, StableDiffusionXLPipeline):
        models = [pipeline.unet, pipeline.text_encoder, pipeline.text_encoder_2]
        lora_prefixes = ["lora_unet_", "lora_te1_", "lora_te2_"]
    else:
        raise ValueError(f"Unexpected pipeline type: {type(pipeline)}")

    # Although we are not unpatching, the patcher might require this. Initialize empty.
    original_weights = OriginalWeightsStorage()

    for lora_model_path, lora_model_weight in lora_models:
        # Load state dict from file
        lora_path = Path(lora_model_path)
        if lora_path.suffix == ".safetensors":
            state_dict = load_file(lora_path.absolute().as_posix(), device="cpu")
        else:
            # Assuming .ckpt, .pt, .bin etc. are torch checkpoints
            state_dict = torch.load(lora_path, map_location="cpu")

        # Convert state dict to ModelPatchRaw
        lora_model = lora_model_from_sd_state_dict(state_dict=state_dict)

        # Apply the patch using LayerPatcher
        for model, lora_prefix in zip(models, lora_prefixes, strict=True):
            LayerPatcher.apply_smart_model_patch(
                model=model,
                prefix=lora_prefix,
                patch=lora_model,
                patch_weight=lora_model_weight,
                original_weights=original_weights,  # Pass storage, even if unused for merging
                original_modules={},  # Pass empty dict, not needed for direct patching/merging
                dtype=model.dtype,  # Use the model's dtype
                # Force direct patching since we are merging into the main weights
                force_direct_patching=True,
                force_sidecar_patching=False,
            )
        logger.info(f"Applied LoRA model '{lora_model_path}' with weight {lora_model_weight}.")

    output_path = Path(output)
    output_path.mkdir(parents=True)

    # TODO(ryand): Should we keep the base model variant? This is clearly a flawed assumption.
    pipeline.save_pretrained(output_path, variant=base_model_variant)
    logger.info(f"Saved merged model to '{output_path}'.")


def parse_lora_model_arg(lora_model_arg: str) -> tuple[str, float]:
    """Parse a --lora-model argument into a tuple of the model path and weight."""
    parts = lora_model_arg.split("::")
    if len(parts) == 1:
        return parts[0], 1.0
    elif len(parts) == 2:
        return parts[0], float(parts[1])
    else:
        raise ValueError(f"Unexpected format for --lora-model arg: '{lora_model_arg}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["SD", "SDXL"],
        help="The type of the models to merge ['SD', 'SDXL'].",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="The base model to merge LoRAs into. The model can be either 1) an HF hub name, 2) a path to a local "
        "diffusers model directory, or 3) a path to a single checkpoint file. An HF variant can optionally be appended "
        "to the model name after a double-colon delimiter ('::')."
        "E.g. '--base-model runwayml/stable-diffusion-v1-5::fp16'",
        required=True,
    )
    parser.add_argument(
        "--lora-models",
        type=str,
        nargs="+",
        help="The path(s) to one or more LoRA models to merge into the base model. Model weights can be appended to "
        "the path, separated by a double colon ('::'). The weight is optional and defaults to 1.0. E.g. "
        "'--lora-models path/to/lora_model_1.safetensors::0.5 path/to/lora_model_2.safetensors'.",
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
    logger = logging.getLogger()

    base_model, base_model_variant = parse_model_arg(args.base_model)
    lora_models = [parse_lora_model_arg(arg) for arg in args.lora_models]

    # Log the parsed arguments
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Base model variant: {base_model_variant}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Save dtype: {args.save_dtype}")
    lora_models_str = "  - " + "\n  - ".join([f"{model} ({weight})" for model, weight in lora_models])
    logger.info(f"LoRA models:\n{lora_models_str}")

    merge_lora_into_sd_model(
        logger=logger,
        model_type=PipelineVersionEnum(args.model_type),
        base_model=base_model,
        base_model_variant=base_model_variant,
        lora_models=lora_models,
        output=args.output,
        save_dtype=args.save_dtype,
    )


if __name__ == "__main__":
    main()

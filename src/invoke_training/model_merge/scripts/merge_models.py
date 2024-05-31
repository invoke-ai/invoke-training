import argparse
import logging
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

from invoke_training._shared.stable_diffusion.model_loading_utils import PipelineVersionEnum, load_pipeline
from invoke_training.model_merge.merge_models import merge_models
from invoke_training.scripts._experimental.lora_merge.merge_lora_into_sd_model import str_to_dtype


def run_merge_models(
    logger: logging.Logger,
    model_type: PipelineVersionEnum,
    models: list[str],
    variants: list[str | None],
    weights: list[float],
    method: str,
    out_dir: str,
    dtype: torch.dtype,
):
    # Create the output directory if it doesn't exist.
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=False)

    # Load the models.
    loaded_models: list[StableDiffusionPipeline] | list[StableDiffusionXLPipeline] = []
    for model, variant in zip(models, variants, strict=True):
        loaded_model = load_pipeline(
            logger=logger,
            model_name_or_path=model,
            pipeline_version=model_type,
            torch_dtype=dtype,
            variant=variant,
        )
        loaded_models.append(loaded_model)

    # Select the submodels to merge.
    if model_type == PipelineVersionEnum.SDXL:
        submodel_names = ["unet", "text_encoder", "text_encoder_2"]
    elif model_type == PipelineVersionEnum.SD:
        submodel_names = ["unet", "text_encoder"]
    else:
        raise ValueError(f"Unexpected model type: {model_type}")

    # Merge the models.
    for submodel_name in submodel_names:
        submodels: list[torch.nn.Module] = [getattr(loaded_model, submodel_name) for loaded_model in loaded_models]
        submodel_state_dicts: list[dict[str, torch.Tensor]] = [submodel.state_dict() for submodel in submodels]

        merged_state_dict = merge_models(state_dicts=submodel_state_dicts, weights=weights, merge_method=method)

        # Merge the merged_state_dict back into the first pipeline to keep memory utilization low.
        submodels[0].load_state_dict(merged_state_dict, assign=True)
        logger.info(f"Merged {submodel_name} state_dicts.")

    # Save the merged model.
    logger.info("Saving result...")
    loaded_models[0].save_pretrained(out_dir_path)
    logger.info(f"Saved merged model to '{out_dir_path}'.")


def parse_models_arg(models: list[str]) -> tuple[list[str], list[str | None]]:
    """Parse a --models argument into models and variants."""
    out_models: list[str] = []
    out_variants: list[str | None] = []
    for model in models:
        parts = model.split("::")
        if len(parts) == 1:
            out_models.append(parts[0])
            out_variants.append(None)
        elif len(parts) == 2:
            out_models.append(parts[0])
            out_variants.append(parts[1])
        else:
            raise ValueError(f"Unexpected format for --models arg: '{model}'.")
    return out_models, out_variants


def main():
    parser = argparse.ArgumentParser()

    # TODO(ryand): Auto-detect the base-model-type.
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["SD", "SDXL"],
        help="The type of the models to merge ['SD', 'SDXL'].",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        required=True,
        help="Two or more models to merge. Each model can be either 1) an HF hub name, 2) a path to a local diffusers "
        "model directory, or 3) a path to a single checkpoint file. An HF variant can optionally be appended to the "
        "model name after a double-colon delimiter ('::')."
        "E.g. '--models runwayml/stable-diffusion-v1-5::fp16 path/to/local/model.safetensors'",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        required=True,
        help="The weights for each model. The weights will be normalized to sum to 1. "
        "For example, to merge weights with equal weights: '--weights 1.0 1.0'. "
        "To weight the first model more heavily: '--weights 0.75 0.25'.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="LERP",
        choices=["LERP", "SLERP"],
        help="The merge method to use. Options: 'LERP' (linear interpolation) or 'SLERP' (spherical linear "
        "interpolation).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="The output directory where the merged model will be written (in diffusers format).",
    )
    parser.add_argument(
        "--dtype",
        help="The torch dtype that will be used for all calculations and for the output model.",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    models, variants = parse_models_arg(args.models)
    run_merge_models(
        logger=logger,
        model_type=PipelineVersionEnum(args.model_type),
        models=models,
        variants=variants,
        weights=[float(w) for w in args.weights],
        method=args.method,
        out_dir=args.out_dir,
        dtype=str_to_dtype(args.dtype),
    )


if __name__ == "__main__":
    main()

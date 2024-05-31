import argparse
import logging
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

from invoke_training._shared.stable_diffusion.model_loading_utils import PipelineVersionEnum, load_pipeline
from invoke_training.model_merge.merge_tasks_to_base import merge_tasks_to_base_model
from invoke_training.model_merge.scripts.merge_models import MergeModel, parse_model_args
from invoke_training.scripts._experimental.lora_merge.merge_lora_into_sd_model import str_to_dtype


def run_merge_models(
    logger: logging.Logger,
    model_type: PipelineVersionEnum,
    base_model: MergeModel,
    task_models: list[MergeModel],
    method: str,
    out_dir: str,
    dtype: torch.dtype,
):
    # Create the output directory if it doesn't exist.
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=False)

    # Load the base model.
    loaded_base_model = load_pipeline(
        logger=logger,
        model_name_or_path=base_model.model_name_or_path,
        pipeline_version=model_type,
        torch_dtype=dtype,
        variant=base_model.variant,
    )

    # Load the task models.
    loaded_task_models: list[StableDiffusionPipeline] | list[StableDiffusionXLPipeline] = []
    for task_model in task_models:
        loaded_task_model = load_pipeline(
            logger=logger,
            model_name_or_path=task_model.model_name_or_path,
            pipeline_version=model_type,
            torch_dtype=dtype,
            variant=task_model.variant,
        )
        loaded_task_models.append(loaded_task_model)

    # Select the submodels to merge.
    if model_type == PipelineVersionEnum.SDXL:
        submodel_names = ["unet", "text_encoder", "text_encoder_2"]
    elif model_type == PipelineVersionEnum.SD:
        submodel_names = ["unet", "text_encoder"]
    else:
        raise ValueError(f"Unexpected model type: {model_type}")

    # Merge the models.
    task_model_weights = [task_model.weight for task_model in task_models]
    for submodel_name in submodel_names:
        base_submodel: torch.nn.Module = getattr(loaded_base_model, submodel_name)
        base_submodel_state_dict = base_submodel.state_dict()
        task_submodels: list[torch.nn.Module] = [
            getattr(loaded_task_model, submodel_name) for loaded_task_model in loaded_task_models
        ]
        task_submodel_state_dict = [submodel.state_dict() for submodel in task_submodels]

        merged_state_dict = merge_tasks_to_base_model(
            base_state_dict=base_submodel_state_dict,
            task_state_dicts=task_submodel_state_dict,
            task_weights=task_model_weights,
            merge_method=method,
        )

        # Merge the merged_state_dict back into the base model pipeline to keep memory utilization low.
        base_submodel.load_state_dict(merged_state_dict, assign=True)
        logger.info(f"Merged {submodel_name} state_dicts.")

    # Save the merged model.
    logger.info("Saving result...")
    loaded_base_model.save_pretrained(out_dir_path)
    logger.info(f"Saved merged model to '{out_dir_path}'.")


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
        "--base-model",
        type=str,
        help="The base model to merge task-specific models into. Can be either 1) an HF hub name, 2) a path to a local "
        "diffusers model directory, or 3) a path to a single checkpoint file. An HF variant can optionally be appended "
        "to the model name after a double-colon delimiter ('::')."
        "E.g. '--base-model runwayml/stable-diffusion-v1-5::fp16'.",
    )
    parser.add_argument(
        "--task-models",
        nargs="+",
        type=str,
        required=True,
        help="One or more task-specific models to merge into the base model. Each model can be either 1) an HF hub "
        "name, 2) a path to a local diffusers model directory, or 3) a path to a single checkpoint file. An HF variant "
        "can optionally be appended to the model name after a double-colon delimiter ('::')."
        "E.g. '--task-models runwayml/stable-diffusion-v1-5::fp16 path/to/local/model.safetensors'",
    )
    parser.add_argument(
        "--task-weights",
        nargs="+",
        type=float,
        required=True,
        help="The weights for each task model. The weights are multipliers applied to the diff between each task model "
        "and the base model. As a starting point, it is recommended to use a weight of 1.0 for all task models, e.g. "
        "'--task-weights 1.0 1.0'. The weights can then be tuned from there, e.g. '--task-weights 1.0 1.3'.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="TIES",
        choices=["TIES", "DARE_LINEAR", "DARE_TIES"],
        help="The merge method to use. Options: ['TIES', 'DARE_LINEAR', 'DARE_TIES'].",
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

    base_model = parse_model_args([args.base_model], [1.0])[0]
    task_models = parse_model_args(args.task_models, args.task_weights)
    run_merge_models(
        logger=logger,
        model_type=PipelineVersionEnum(args.model_type),
        base_model=base_model,
        task_models=task_models,
        method=args.method,
        out_dir=args.out_dir,
        dtype=str_to_dtype(args.dtype),
    )


if __name__ == "__main__":
    main()

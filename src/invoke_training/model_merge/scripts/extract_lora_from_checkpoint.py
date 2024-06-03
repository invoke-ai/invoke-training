# This script is based on
# https://raw.githubusercontent.com/kohya-ss/sd-scripts/bfb352bc433326a77aca3124248331eb60c49e8c/networks/extract_lora_from_models.py
# That script was originally based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import peft
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from invoke_training._shared.accelerator.accelerator_utils import get_dtype_from_str
from invoke_training._shared.stable_diffusion.lora_checkpoint_utils import (
    UNET_TARGET_MODULES,
    save_sdxl_kohya_checkpoint,
)
from invoke_training._shared.stable_diffusion.model_loading_utils import PipelineVersionEnum, load_pipeline
from invoke_training.model_merge.extract_lora import (
    PEFT_BASE_LAYER_PREFIX,
    extract_lora_from_diffs,
    get_patched_base_weights_from_peft_model,
    get_state_dict_diff,
)
from invoke_training.model_merge.utils.parse_model_arg import parse_model_arg


@dataclass
class StableDiffusionModel:
    """A helper class to store the submodels of a SD model that we are interested in for LoRA extraction."""

    unet: UNet2DConditionModel | None = None
    # TODO(ryand): Figure out the actual type of these text encoders.
    text_encoder: CLIPTextModel | None = None
    text_encoder_2: CLIPTextModelWithProjection | None = None

    def all_none(self) -> bool:
        return self.unet is None and self.text_encoder is None and self.text_encoder_2 is None


def load_model(
    logger: logging.Logger,
    model_name_or_path: str,
    model_type: PipelineVersionEnum,
    variant: str | None,
    dtype: torch.dtype,
) -> StableDiffusionModel:
    sd_model = StableDiffusionModel()

    model_path = Path(model_name_or_path)
    if model_path.is_dir():
        # model_path is a directory, so we'll try to load the submodels of interest from its subdirectories.
        logger.info(f"'{model_name_or_path}' is a directory. Attempting to load submodels.")
        for submodel_name, submodel_class in [
            ("unet", UNet2DConditionModel),
            ("text_encoder", CLIPTextModel),
            ("text_encoder_2", CLIPTextModelWithProjection),
        ]:
            submodel_path: Path = model_path / submodel_name
            if submodel_path.exists():
                logger.info(f"Loading '{submodel_name}' from '{submodel_path}'.")
                # TODO(ryand): Add variant fallbacks?
                submodel = submodel_class.from_pretrained(
                    submodel_path, variant=variant, torch_dtype=dtype, local_files_only=True
                )
                setattr(sd_model, submodel_name, submodel)
            else:
                logger.info(f"'{submodel_name}' not found in '{model_name_or_path}'. Skipping.")
                continue
    else:
        # model_name_or_path is not a directory, so it is either:
        # 1) a single checkpoint file
        # 2) a HF model name
        # Both can be loaded by calling load_pipeline.
        logger.info(f"'{model_name_or_path}' is a single checkpoint file. Attempting to load.")
        pipeline = load_pipeline(
            logger=logger,
            model_name_or_path=model_name_or_path,
            pipeline_version=model_type,
            torch_dtype=dtype,
            variant=variant,
        )
        if isinstance(pipeline, StableDiffusionPipeline):
            sd_model.unet = pipeline.unet
            sd_model.text_encoder = pipeline.text_encoder
        elif isinstance(pipeline, StableDiffusionXLPipeline):
            sd_model.unet = pipeline.unet
            sd_model.text_encoder = pipeline.text_encoder
            sd_model.text_encoder_2 = pipeline.text_encoder_2
        else:
            raise RuntimeError(f"Unexpected pipeline type: {type(pipeline)}.")

    if sd_model.all_none():
        raise RuntimeError(f"Failed to load any submodels from '{model_name_or_path}'.")

    return sd_model


def str_to_device(device_str: Literal["cuda", "cpu"]) -> torch.device:
    if device_str == "cuda":
        return torch.device("cuda")
    elif device_str == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Unexpected device: {device_str}")


# TODO(ryand): Delete this after integrating the variant fallback logic.
# def load_sdxl_unet(model_path: str) -> UNet2DConditionModel:
#     variants_to_try = [None, "fp16"]
#     unet = None
#     for variant in variants_to_try:
#         try:
#             unet = UNet2DConditionModel.from_pretrained(model_path, variant=variant, local_files_only=True)
#         except OSError as e:
#             if "no file named" in str(e):
#                 # Ok. We'll try a different variant.
#                 pass
#             else:
#                 raise
#     if unet is None:
#         raise RuntimeError(f"Failed to load UNet from '{model_path}'.")
#     return unet


def state_dict_to_device(state_dict: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device=device) for k, v in state_dict.items()}


def extract_lora_from_submodel(
    logger: logging.Logger,
    model_orig: torch.nn.Module,
    model_tuned: torch.nn.Module,
    device: torch.device,
    out_dtype: torch.dtype,
    lora_rank: int,
    clamp_quantile: float = 0.99,
) -> peft.PeftModel:
    """Extract LoRA weights from the diff between model_orig and model_tuned. Returns a new model_orig, wrapped in a
    PeftModel, with the LoRA weights applied.
    """
    # Apply LoRA to the UNet.
    # The only reason we do this is to get the module names for the weights that we'll extract. We don't actually use
    # the LoRA weights initialized here.
    unet_lora_config = peft.LoraConfig(
        r=lora_rank,
        # We set the alpha to the rank, because we don't want any scaling to be applied to the LoRA weights that we
        # extract.
        lora_alpha=lora_rank,
        target_modules=UNET_TARGET_MODULES,
    )
    model_tuned = peft.get_peft_model(model_tuned, unet_lora_config)
    model_orig = peft.get_peft_model(model_orig, unet_lora_config)

    base_weights_tuned = get_patched_base_weights_from_peft_model(model_tuned)
    base_weights_orig = get_patched_base_weights_from_peft_model(model_orig)

    diffs = get_state_dict_diff(base_weights_tuned, base_weights_orig)

    # Clear tuned model to save memory.
    # TODO(ryand): We also need to clear the state_dicts. Move the diff extraction to a separate function so that memory
    # cleanup is handled by scoping.
    del model_tuned

    # Apply SVD (Singluar Value Decomposition) to the diffs.
    # We just use the device for this calculation, since it's slow, then we move the results back to the CPU.
    logger.info("Calculating LoRA weights with SVD.")
    diffs = state_dict_to_device(diffs, device)
    lora_weights = extract_lora_from_diffs(
        diffs=diffs, rank=lora_rank, clamp_quantile=clamp_quantile, out_dtype=out_dtype
    )

    # Prepare state dict for LoRA.
    lora_state_dict = {}
    for module_name, (lora_up, lora_down) in lora_weights.items():
        lora_state_dict[PEFT_BASE_LAYER_PREFIX + module_name + ".lora_A.default.weight"] = lora_down
        lora_state_dict[PEFT_BASE_LAYER_PREFIX + module_name + ".lora_B.default.weight"] = lora_up
        # The alpha value is set once globally in the PEFT model, so no need to set it for each module.
        # lora_state_dict[peft_base_layer_suffix + module_name + ".alpha"] = torch.tensor(down_weight.size()[0])

    lora_state_dict = state_dict_to_device(lora_state_dict, torch.device("cpu"))

    # Load the state_dict into the LoRA model.
    model_orig.load_state_dict(lora_state_dict, strict=False, assign=True)

    return model_orig


@torch.no_grad()
def extract_lora(
    logger: logging.Logger,
    model_type: PipelineVersionEnum,
    orig_model_name_or_path: str,
    orig_model_variant: str | None,
    tuned_model_name_or_path: str,
    tuned_model_variant: str | None,
    save_to: str,
    load_precision: Literal["float32", "float16", "bfloat16"],
    save_precision: Literal["float32", "float16", "bfloat16"],
    device: Literal["cuda", "cpu"],
    lora_rank: int,
    clamp_quantile=0.99,
):
    load_dtype = get_dtype_from_str(load_precision)
    save_dtype = get_dtype_from_str(save_precision)
    device = str_to_device(device)

    # Load models.
    # if model_type == "sd1":
    #     raise NotImplementedError("SD1 support is not yet implemented.")
    # elif model_type == "sdxl":
    #     logger.info(f"Loading original SDXL model: '{model_orig_path}'.")
    #     unet_orig = load_sdxl_unet(model_orig_path)
    #     logger.info(f"Loading tuned SDXL model: '{model_tuned_path}'.")
    #     unet_tuned = load_sdxl_unet(model_tuned_path)

    #     if load_dtype is not None:
    #         unet_orig = unet_orig.to(load_dtype)
    #         unet_tuned = unet_tuned.to(load_dtype)
    # else:
    #     raise ValueError(f"Unexpected model type: '{model_type}'.")

    orig_model = load_model(
        logger=logger,
        model_name_or_path=orig_model_name_or_path,
        model_type=model_type,
        dtype=load_dtype,
        variant=orig_model_variant,
    )
    tuned_model = load_model(
        logger=logger,
        model_name_or_path=tuned_model_name_or_path,
        model_type=model_type,
        dtype=load_dtype,
        variant=tuned_model_variant,
    )

    # TODO(ryand): Consolidate these calls to extract_lora_from_submodel.
    unet_orig_with_lora = None
    if orig_model.unet is not None and tuned_model.unet is not None:
        logger.info("Extracting LoRA from UNet.")
        unet_orig_with_lora = extract_lora_from_submodel(
            logger=logger,
            model_orig=orig_model.unet,
            model_tuned=tuned_model.unet,
            device=device,
            out_dtype=save_dtype,
            lora_rank=lora_rank,
            clamp_quantile=clamp_quantile,
        )

    text_encoder_orig_with_lora = None
    if orig_model.text_encoder is not None and tuned_model.text_encoder is not None:
        logger.info("Extracting LoRA from text encoder.")
        text_encoder_orig_with_lora = extract_lora_from_submodel(
            logger=logger,
            model_orig=orig_model.text_encoder,
            model_tuned=tuned_model.text_encoder,
            device=device,
            out_dtype=save_dtype,
            lora_rank=lora_rank,
            clamp_quantile=clamp_quantile,
        )

    text_encoder_2_orig_with_lora = None
    if orig_model.text_encoder_2 is not None and tuned_model.text_encoder_2 is not None:
        logger.info("Extracting LoRA from text encoder 2.")
        text_encoder_2_orig_with_lora = extract_lora_from_submodel(
            logger=logger,
            model_orig=orig_model.text_encoder_2,
            model_tuned=tuned_model.text_encoder_2,
            device=device,
            out_dtype=save_dtype,
            lora_rank=lora_rank,
            clamp_quantile=clamp_quantile,
        )

    # Save the LoRA weights.
    save_to_path = Path(save_to)
    assert save_to_path.suffix == ".safetensors"
    if save_to_path.exists():
        raise FileExistsError(f"Destination file already exists: '{save_to}'.")
    save_to_path.parent.mkdir(parents=True, exist_ok=True)
    save_sdxl_kohya_checkpoint(
        save_to_path,
        unet=unet_orig_with_lora,
        text_encoder_1=text_encoder_orig_with_lora,
        text_encoder_2=text_encoder_2_orig_with_lora,
    )

    logger.info(f"Saved LoRA weights to: {save_to_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["SD", "SDXL"],
        help="The type of the models to merge ['SD', 'SDXL'].",
    )
    parser.add_argument(
        "--model-orig",
        type=str,
        required=True,
        help="Path or HF Hub name of the original model. The model must be in one of the following formats: "
        "1) a single checkpoint file (e.g. '.safetensors') containing all submodels, "
        "2) a model in diffusers format containing all submodels, "
        "or 3) a model in diffusers format containing a subset of the submodels (e.g. only a UNet)."
        "An HF variant can optionally be appended to the model name after a double-colon delimiter ('::')."
        "E.g. '--model-orig runwayml/stable-diffusion-v1-5::fp16'",
    )
    parser.add_argument(
        "--model-tuned",
        type=str,
        required=True,
        help="Path or HF Hub name of the tuned model. The model must be in one of the following formats: "
        "1) a single checkpoint file (e.g. '.safetensors') containing all submodels, "
        "2) a model in diffusers format containing all submodels, "
        "or 3) a model in diffusers format containing a subset of the submodels (e.g. only a UNet)."
        "An HF variant can optionally be appended to the model name after a double-colon delimiter ('::')."
        "E.g. '--model-orig runwayml/stable-diffusion-v1-5::fp16'",
    )
    parser.add_argument(
        "--save-to",
        type=str,
        required=True,
        help="Destination file path (must have a .safetensors extension).",
    )
    parser.add_argument(
        "--load-precision",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model load precision.",
    )
    parser.add_argument(
        "--save-precision",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Model save precision.",
    )

    parser.add_argument("--lora-rank", type=int, default=4, help="LoRA rank dimension.")
    parser.add_argument("--clamp-quantile", type=float, default=0.99, help="Quantile clamping value. (0-1)")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use. (cuda or cpu)"
    )

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    orig_model_name_or_path, orig_model_variant = parse_model_arg(args.model_orig)
    tuned_model_name_or_path, tuned_model_variant = parse_model_arg(args.model_tuned)

    extract_lora(
        logger=logger,
        model_type=PipelineVersionEnum(args.model_type),
        orig_model_name_or_path=orig_model_name_or_path,
        orig_model_variant=orig_model_variant,
        tuned_model_name_or_path=tuned_model_name_or_path,
        tuned_model_variant=tuned_model_variant,
        save_to=args.save_to,
        load_precision=args.load_precision,
        save_precision=args.save_precision,
        device=args.device,
        lora_rank=args.lora_rank,
        clamp_quantile=args.clamp_quantile,
    )


if __name__ == "__main__":
    main()

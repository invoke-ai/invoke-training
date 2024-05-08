# This script is based on
# https://raw.githubusercontent.com/kohya-ss/sd-scripts/bfb352bc433326a77aca3124248331eb60c49e8c/networks/extract_lora_from_models.py
# That script was originally based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py


import argparse
import logging
import os
from pathlib import Path
from typing import Literal

import peft
import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file
from tqdm import tqdm

from invoke_training._shared.stable_diffusion.lora_checkpoint_utils import (
    UNET_TARGET_MODULES,
    save_sdxl_kohya_checkpoint,
)


def save_to_file(file_name, model, state_dict, dtype):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if type(state_dict[key]) == torch.Tensor:
                state_dict[key] = state_dict[key].to(dtype)

    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(model, file_name)
    else:
        torch.save(model, file_name)


def str_to_dtype(dtype_str: Literal["fp32", "fp16", "bf16"]):
    if dtype_str == "fp32":
        return torch.float32
    elif dtype_str == "fp16":
        return torch.float16
    elif dtype_str == "bf16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unexpected dtype: {dtype_str}")


def load_sdxl_unet(model_path: str) -> UNet2DConditionModel:
    variants_to_try = [None, "fp16"]
    unet = None
    for variant in variants_to_try:
        try:
            unet = UNet2DConditionModel.from_pretrained(model_path, variant=variant, local_files_only=True)
        except OSError as e:
            if "no file named" in str(e):
                # Ok. We'll try a different variant.
                pass
            else:
                raise
    if unet is None:
        raise RuntimeError(f"Failed to load UNet from '{model_path}'.")
    return unet


@torch.no_grad()
def extract_lora_from_diffs(
    diffs: dict[str, torch.Tensor], rank: int, clamp_quantile: float, out_dtype: torch.dtype
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    lora_weights = {}
    for lora_name, mat in tqdm(list(diffs.items())):
        # Use full precision for the intermediate calculations.
        mat = mat.to(torch.float32)

        is_conv2d = len(mat.size()) == 4
        if is_conv2d:
            # TODO(ryand)
            raise NotImplementedError("Conv2D support is not yet implemented.")

        out_dim, in_dim = mat.size()[0:2]

        # LoRA rank cannot exceed the original dimensions.
        assert rank < in_dim
        assert rank < out_dim

        u: torch.Tensor
        s: torch.Tensor
        v_h: torch.Tensor
        u, s, v_h = torch.linalg.svd(mat)

        # Apply the Eckart-Young-Mirsky theorem.
        # https://en.wikipedia.org/wiki/Low-rank_approximation#Proof_of_Eckart%E2%80%93Young%E2%80%93Mirsky_theorem_(for_Frobenius_norm)
        u = u[:, :rank]
        s = s[:rank]
        u = u @ torch.diag(s)

        v_h = v_h[:rank, :]

        # At this point, u is the lora_up (a.k.a. lora_B) weight, and v_h is the lora_down (a.k.a. lora_A) weight.
        # The reason we don't use more appropriate variable names is to keep memory usage low - we want the old tensors
        # to get cleaned up after each operation.

        # Clamp the outliers.
        dist = torch.cat([u.flatten(), v_h.flatten()])
        hi_val = torch.quantile(dist, clamp_quantile)
        low_val = -hi_val

        u = u.clamp(low_val, hi_val)
        v_h = v_h.clamp(low_val, hi_val)

        u = u.to(dtype=out_dtype).contiguous()
        v_h = v_h.to(dtype=out_dtype).contiguous()

        lora_weights[lora_name] = (u, v_h)
    return lora_weights


@torch.no_grad()
def svd(
    logger: logging.Logger,
    model_type: Literal["sd1", "sdxl"],
    model_orig_path: str,
    model_tuned_path: str,
    save_to: str,
    load_precision: Literal["fp32", "fp16", "bf16"],
    save_precision: Literal["fp32", "fp16", "bf16"],
    lora_rank: int,
    clamp_quantile=0.99,
):
    load_dtype = str_to_dtype(load_precision)
    save_dtype = str_to_dtype(save_precision)

    # Load models.
    if model_type == "sd1":
        raise NotImplementedError("SD1 support is not yet implemented.")
    elif model_type == "sdxl":
        logger.info(f"Loading original SDXL model: '{model_orig_path}'.")
        unet_orig = load_sdxl_unet(model_orig_path)
        logger.info(f"Loading tuned SDXL model: '{model_tuned_path}'.")
        unet_tuned = load_sdxl_unet(model_tuned_path)

        if load_dtype is not None:
            unet_orig = unet_orig.to(load_dtype)
            unet_tuned = unet_tuned.to(load_dtype)
    else:
        raise ValueError(f"Unexpected model type: '{model_type}'.")

    # Apply LoRA to the UNet.
    # The only reason we do this is to get the module names for the weights that we'll extract. We don't actually use
    # the LoRA weights initialized here.
    unet_lora_config = peft.LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=UNET_TARGET_MODULES,
    )
    unet_tuned = peft.get_peft_model(unet_tuned, unet_lora_config)
    unet_orig = peft.get_peft_model(unet_orig, unet_lora_config)

    diffs: dict[str, torch.Tensor] = {}
    state_dict_tuned = unet_tuned.state_dict()
    state_dict_orig = unet_orig.state_dict()
    peft_base_layer_suffix = ".base_layer.weight"
    peft_base_layer_prefix = "base_model.model."
    for weight_name in state_dict_tuned:
        # Weights that end with ".base_layer.weight" are the original weights for LoRA layers.
        if weight_name.endswith(peft_base_layer_suffix):
            # Extract the base module name.
            module_name = weight_name[: -len(peft_base_layer_suffix)]
            assert module_name.startswith(peft_base_layer_prefix)
            module_name = module_name[len(peft_base_layer_prefix) :]

            diffs[module_name] = state_dict_tuned[weight_name] - state_dict_orig[weight_name]

    # Clear tuned UNet to save memory.
    # TODO(ryand): We also need to clear the state_dicts. Move the diff extraction to a separate function so that memory
    # cleanup is handled by scoping.
    del unet_tuned

    # Apply SVD (Singluar Value Decomposition) to the diffs.
    logger.info("Calculating LoRA weights with SVD.")
    lora_weights = extract_lora_from_diffs(
        diffs=diffs, rank=lora_rank, clamp_quantile=clamp_quantile, out_dtype=save_dtype
    )

    # Prepare state dict for LoRA.
    lora_state_dict = {}
    for module_name, (lora_up, lora_down) in lora_weights.items():
        lora_state_dict[peft_base_layer_prefix + module_name + ".lora_A.default.weight"] = lora_down
        lora_state_dict[peft_base_layer_prefix + module_name + ".lora_B.default.weight"] = lora_up
        # TODO(ryand): Double-check that this isn't needed with peft.
        # lora_state_dict[peft_base_layer_suffix + module_name + ".alpha"] = torch.tensor(down_weight.size()[0])

    # Load the state_dict into the LoRA model.
    unet_orig.load_state_dict(lora_state_dict, strict=False, assign=True)

    save_to_path = Path(save_to)
    assert save_to_path.suffix == ".safetensors"
    if save_to_path.exists():
        raise FileExistsError(f"Destination file already exists: '{save_to}'.")
    save_to_path.parent.mkdir(parents=True, exist_ok=True)
    save_sdxl_kohya_checkpoint(save_to_path, unet=unet_orig, text_encoder_1=None, text_encoder_2=None)

    logger.info(f"Saved LoRA weights to: {save_to_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="sd1", choices=["sd1", "sdxl"], help="The base model type.")

    parser.add_argument("--model-orig", type=str, required=True, help="Path to the original model.")
    parser.add_argument("--model-tuned", type=str, required=True, help="Path to the tuned model.")
    parser.add_argument(
        "--save-to",
        type=str,
        required=True,
        help="Destination file path (must have a .safetensors extension).",
    )
    parser.add_argument(
        "--load-precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Model load precision."
    )
    parser.add_argument(
        "--save-precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Model save precision."
    )

    parser.add_argument("--lora-rank", type=int, default=4, help="LoRA rank dimension.")
    parser.add_argument("--clamp-quantile", type=float, default=0.99, help="Quantile clamping value. (0-1)")

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    svd(
        logger=logger,
        model_type=args.model_type,
        model_orig_path=args.model_orig,
        model_tuned_path=args.model_tuned,
        save_to=args.save_to,
        load_precision=args.load_precision,
        save_precision=args.save_precision,
        lora_rank=args.lora_rank,
        clamp_quantile=args.clamp_quantile,
    )


if __name__ == "__main__":
    main()

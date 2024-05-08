# This script is based on
# https://raw.githubusercontent.com/kohya-ss/sd-scripts/bfb352bc433326a77aca3124248331eb60c49e8c/networks/extract_lora_from_models.py
# That script was originally based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py


import argparse
import logging
import os
from typing import Literal

import peft
import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file
from tqdm import tqdm

from invoke_training._shared.stable_diffusion.lora_checkpoint_utils import UNET_TARGET_MODULES


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
    device = torch.cpu
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
        # TODO(ryand): Should I set this to lora_rank?
        lora_alpha=1.0,
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
    logger.info("calculating by svd")
    lora_weights = {}
    for lora_name, mat in tqdm(list(diffs.items())):
        if args.device:
            mat = mat.to(args.device)
        mat = mat.to(torch.float)  # calc by float

        # if conv_dim is None, diffs do not include LoRAs for conv2d-3x3
        conv2d = len(mat.size()) == 4
        kernel_size = None if not conv2d else mat.size()[2:4]
        conv2d_3x3 = conv2d and kernel_size != (1, 1)

        rank = dim if not conv2d_3x3 or conv_dim is None else conv_dim
        out_dim, in_dim = mat.size()[0:2]

        if device:
            mat = mat.to(device)

        # logger.info(lora_name, mat.size(), mat.device, rank, in_dim, out_dim)
        rank = min(rank, in_dim, out_dim)  # LoRA rank cannot exceed the original dim

        if conv2d:
            if conv2d_3x3:
                mat = mat.flatten(start_dim=1)
            else:
                mat = mat.squeeze()

        U, S, Vh = torch.linalg.svd(mat)

        U = U[:, :rank]
        S = S[:rank]
        U = U @ torch.diag(S)

        Vh = Vh[:rank, :]

        dist = torch.cat([U.flatten(), Vh.flatten()])
        hi_val = torch.quantile(dist, clamp_quantile)
        low_val = -hi_val

        U = U.clamp(low_val, hi_val)
        Vh = Vh.clamp(low_val, hi_val)

        if conv2d:
            U = U.reshape(out_dim, rank, 1, 1)
            Vh = Vh.reshape(rank, in_dim, kernel_size[0], kernel_size[1])

        U = U.to(work_device, dtype=save_dtype).contiguous()
        Vh = Vh.to(work_device, dtype=save_dtype).contiguous()

        lora_weights[lora_name] = (U, Vh)

    # make state dict for LoRA
    lora_sd = {}
    for lora_name, (up_weight, down_weight) in lora_weights.items():
        lora_sd[lora_name + ".lora_up.weight"] = up_weight
        lora_sd[lora_name + ".lora_down.weight"] = down_weight
        lora_sd[lora_name + ".alpha"] = torch.tensor(down_weight.size()[0])

    # load state dict to LoRA and save it
    lora_network_save, lora_sd = lora.create_network_from_weights(
        1.0, None, None, text_encoders_o, unet_o, weights_sd=lora_sd
    )
    lora_network_save.apply_to(text_encoders_o, unet_o)  # create internal module references for state_dict

    info = lora_network_save.load_state_dict(lora_sd)
    logger.info(f"Loading extracted LoRA weights: {info}")

    dir_name = os.path.dirname(save_to)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    lora_network_save.save_weights(save_to, save_dtype, metadata)
    logger.info(f"LoRA weights are saved to: {save_to}")


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

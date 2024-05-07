# This script is based on
# https://raw.githubusercontent.com/kohya-ss/sd-scripts/bfb352bc433326a77aca3124248331eb60c49e8c/networks/extract_lora_from_models.py
# That script was originally based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py


import argparse
import json
import logging
import os
import time
from typing import Literal

import lora
import torch
from diffusers import UNet2DConditionModel
from library import sai_model_spec
from safetensors.torch import save_file
from tqdm import tqdm


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
    unet = UNet2DConditionModel.from_pretrained(model_path, local_files_only=True)
    return unet


def svd(
    logger: logging.Logger,
    model_type: Literal["sd1", "sdxl"],
    model_orig_path: str,
    model_tuned_path: str,
    load_precision: Literal["fp32", "fp16", "bf16"],
    save_precision: Literal["fp32", "fp16", "bf16"],
    model_org=None,
    model_tuned=None,
    save_to=None,
    dim=4,
    conv_dim=None,
    device=None,
    clamp_quantile=0.99,
    min_diff=0.01,
    no_metadata=False,
):
    load_dtype = str_to_dtype(load_precision)
    save_dtype = str_to_dtype(save_precision)
    work_device = "cpu"

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

    # create LoRA network to extract weights: Use dim (rank) as alpha
    if conv_dim is None:
        kwargs = {}
    else:
        kwargs = {"conv_dim": conv_dim, "conv_alpha": conv_dim}

    lora_network_o = lora.create_network(1.0, dim, dim, None, text_encoders_o, unet_o, **kwargs)
    lora_network_t = lora.create_network(1.0, dim, dim, None, text_encoders_t, unet_t, **kwargs)
    assert (
        len(lora_network_o.text_encoder_loras) == len(lora_network_t.text_encoder_loras)
    ), "model version is different (SD1.x vs SD2.x) / それぞれのモデルのバージョンが違います（SD1.xベースとSD2.xベース） "

    # get diffs
    diffs = {}
    text_encoder_different = False
    for i, (lora_o, lora_t) in enumerate(zip(lora_network_o.text_encoder_loras, lora_network_t.text_encoder_loras)):
        lora_name = lora_o.lora_name
        module_o = lora_o.org_module
        module_t = lora_t.org_module
        diff = module_t.weight.to(work_device) - module_o.weight.to(work_device)

        # clear weight to save memory
        module_o.weight = None
        module_t.weight = None

        # Text Encoder might be same
        if not text_encoder_different and torch.max(torch.abs(diff)) > min_diff:
            text_encoder_different = True
            logger.info(f"Text encoder is different. {torch.max(torch.abs(diff))} > {min_diff}")

        diffs[lora_name] = diff

    # clear target Text Encoder to save memory
    for text_encoder in text_encoders_t:
        del text_encoder

    if not text_encoder_different:
        logger.warning("Text encoder is same. Extract U-Net only.")
        lora_network_o.text_encoder_loras = []
        diffs = {}  # clear diffs

    for i, (lora_o, lora_t) in enumerate(zip(lora_network_o.unet_loras, lora_network_t.unet_loras)):
        lora_name = lora_o.lora_name
        module_o = lora_o.org_module
        module_t = lora_t.org_module
        diff = module_t.weight.to(work_device) - module_o.weight.to(work_device)

        # clear weight to save memory
        module_o.weight = None
        module_t.weight = None

        diffs[lora_name] = diff

    # clear LoRA network, target U-Net to save memory
    del lora_network_o
    del lora_network_t
    del unet_t

    # make LoRA with svd
    logger.info("calculating by svd")
    lora_weights = {}
    with torch.no_grad():
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

    # minimum metadata
    net_kwargs = {}
    if conv_dim is not None:
        net_kwargs["conv_dim"] = str(conv_dim)
        net_kwargs["conv_alpha"] = str(float(conv_dim))

    metadata = {
        "ss_v2": str(v2),
        "ss_base_model_version": model_version,
        "ss_network_module": "networks.lora",
        "ss_network_dim": str(dim),
        "ss_network_alpha": str(float(dim)),
        "ss_network_args": json.dumps(net_kwargs),
    }

    if not no_metadata:
        title = os.path.splitext(os.path.basename(save_to))[0]
        sai_metadata = sai_model_spec.build_metadata(
            None, v2, v_parameterization, sdxl, True, False, time.time(), title=title
        )
        metadata.update(sai_metadata)

    lora_network_save.save_weights(save_to, save_dtype, metadata)
    logger.info(f"LoRA weights are saved to: {save_to}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="sd1", choices=["sd1", "sdxl"], help="The base model type.")
    parser.add_argument(
        "--load-precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Model load precision."
    )
    parser.add_argument(
        "--save-precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Model save precision.",
    )
    parser.add_argument(
        "--model-orig",
        type=str,
        default=None,
        required=True,
        help="Path to the original model.",
    )
    parser.add_argument(
        "--model-tuned",
        type=str,
        default=None,
        required=True,
        help="Path to the tuned model.",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        required=True,
        help="destination file name: ckpt or safetensors file / 保存先のファイル名、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--dim", type=int, default=4, help="dimension (rank) of LoRA (default 4) / LoRAの次元数（rank）（デフォルト4）"
    )
    parser.add_argument(
        "--conv_dim",
        type=int,
        default=None,
        help="dimension (rank) of LoRA for Conv2d-3x3 (default None, disabled) / LoRAのConv2d-3x3の次元数（rank）（デフォルトNone、適用なし）",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="device to use, cuda for GPU / 計算を行うデバイス、cuda でGPUを使う"
    )
    parser.add_argument(
        "--clamp_quantile",
        type=float,
        default=0.99,
        help="Quantile clamping value, float, (0-1). Default = 0.99 / 値をクランプするための分位点、float、(0-1)。デフォルトは0.99",
    )
    parser.add_argument(
        "--min_diff",
        type=float,
        default=0.01,
        help="Minimum difference between finetuned model and base to consider them different enough to extract, float, (0-1). Default = 0.01 /"
        + "LoRAを抽出するために元モデルと派生モデルの差分の最小値、float、(0-1)。デフォルトは0.01",
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save sai modelspec metadata (minimum ss_metadata for LoRA is saved) / "
        + "sai modelspecのメタデータを保存しない（LoRAの最低限のss_metadataは保存される）",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    svd(logger=logger, **vars(args))

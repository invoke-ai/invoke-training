import argparse
from pathlib import Path

import torch

from invoke_training._shared.stable_diffusion.lora_checkpoint_utils import (
    convert_sd_peft_checkpoint_to_kohya_state_dict,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a Stable Diffusion LoRA checkpoint in PEFT format to kohya format."
    )
    parser.add_argument(
        "--src-ckpt-dir",
        type=str,
        required=True,
        help="Path to the source checkpoint directory.",
    )
    parser.add_argument(
        "--dst-ckpt-file",
        type=str,
        required=True,
        help="Path to the destination Kohya checkpoint file.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        help="The precision to save the kohya state dict in. One of ['fp16', 'fp32'].",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    in_checkpoint_dir = Path(args.src_ckpt_dir)
    out_checkpoint_file = Path(args.dst_ckpt_file)

    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unsupported --dtype = '{args.dtype}'.")

    convert_sd_peft_checkpoint_to_kohya_state_dict(
        in_checkpoint_dir=in_checkpoint_dir, out_checkpoint_file=out_checkpoint_file, dtype=dtype
    )

    print(f"Saved kohya checkpoint to '{out_checkpoint_file}'.")


if __name__ == "__main__":
    main()

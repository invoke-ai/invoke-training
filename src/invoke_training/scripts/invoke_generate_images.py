import argparse
from pathlib import Path

from invoke_training.training.shared.model_loading_utils import PipelineVersionEnum
from invoke_training.training.tools.generate_images import generate_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a dataset of images from a single prompt. (Typically used to generate prior "
        "preservation/regularization datasets.)"
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        required=True,
        help="Path to the directory where the images will be stored.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Name or path of the diffusers model to generate images with. Can be in diffusers format, or a single "
        "stable diffusion checkpoint file. (E.g. 'runwayml/stable-diffusion-v1-5', "
        "'stabilityai/stable-diffusion-xl-base-1.0', '/path/to/realisticVisionV51_v51VAE.safetensors', etc. )",
    )
    parser.add_argument(
        "-l",
        "--lora",
        type=str,
        nargs="*",
        help="LoRA models to apply to the base model. The LoRA weight can optionally be provided after a colon "
        "separator. E.g. `-l path/to/lora.bin:0.5 -l path/to/lora_2.safetensors`. ",
    )
    parser.add_argument(
        "--ti",
        type=str,
        nargs="*",
        help="Paths(s) to Textual Inversion embeddings to apply to the base model.",
    )
    parser.add_argument(
        "--sd-version",
        type=str,
        required=True,
        help="The Stable Diffusion version. One of: ['SD', 'SDXL'].",
    )
    parser.add_argument("-p", "--prompt", type=str, required=True, help="The prompt to use for image generation.")
    parser.add_argument(
        "-n",
        "--num-images",
        type=int,
        required=True,
        help="The number of images to generate.",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="The height of the generated images in pixels.",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="The width of the generated images in pixels.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Seed for repeatability.",
    )
    parser.add_argument(
        "--enable-cpu-offload",
        default=False,
        action="store_true",
        help="If True, models will be loaded onto the GPU one by one to conserve VRAM.",
    )
    return parser.parse_args()


def parse_lora_args(lora_args: list[str] | None) -> list[tuple[Path, int]]:
    loras: list[tuple[Path, int]] = []

    lora_args = lora_args or []
    for lora in lora_args:
        lora_split = lora.split(":")

        if len(lora_split) == 1:
            # If weight is not specified, assume 1.0.
            loras.append((Path(lora_split[0]), 1.0))
        elif len(lora_split) == 2:
            loras.append((Path(lora_split[0]), float(lora_split[1])))
        else:
            raise ValueError(f"Invalid lora argument syntax: '{lora}'.")

    return loras


def main():
    args = parse_args()

    loras = parse_lora_args(args.lora)

    print(f"Generating {args.num_images} images in '{args.out_dir}'.")
    generate_images(
        out_dir=args.out_dir,
        model=args.model,
        pipeline_version=PipelineVersionEnum(args.sd_version),
        prompt=args.prompt,
        num_images=args.num_images,
        height=args.height,
        width=args.width,
        loras=loras,
        ti_embeddings=args.ti,
        seed=args.seed,
        enable_cpu_offload=args.enable_cpu_offload,
    )


if __name__ == "__main__":
    main()

import argparse

from invoke_training.training.tools.generate_images import generate_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a dataset of images from a single prompt. (Typically used to generate prior "
        "preservation/regularization datasets.)"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Path to the directory where the images will be stored.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of the diffusers model to generate images with. Can be in diffusers format, or a single "
        "stable diffusion checkpoint file. (E.g. 'runwayml/stable-diffusion-v1-5', "
        "'stabilityai/stable-diffusion-xl-base-1.0', '/path/to/realisticVisionV51_v51VAE.safetensors', etc. )",
    )
    parser.add_argument(
        "--sd-version",
        type=str,
        required=True,
        help="The Stable Diffusion version. One of: ['sd', 'sdxl'].",
    )
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to use for image generation.")
    parser.add_argument(
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


def main():
    args = parse_args()

    print(f"Generating {args.num_images} images in '{args.out_dir}'.")
    generate_images(
        out_dir=args.out_dir,
        model=args.model,
        sd_version=args.sd_version,
        prompt=args.prompt,
        num_images=args.num_images,
        height=args.height,
        width=args.width,
        seed=args.seed,
        enable_cpu_offload=args.enable_cpu_offload,
    )


if __name__ == "__main__":
    main()

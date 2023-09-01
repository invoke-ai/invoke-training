import os

import torch
from tqdm import tqdm

from invoke_training.training.shared.model_loading_utils import (
    PipelineVersionEnum,
    load_pipeline,
)


def generate_images(
    out_dir: str,
    model: str,
    pipeline_version: PipelineVersionEnum,
    prompt: str,
    num_images: int,
    height: int,
    width: int,
    seed: int = 0,
    torch_dtype: torch.dtype = torch.float16,
    torch_device: str = "cuda",
    enable_cpu_offload: bool = False,
):
    """Generate a set of images and store them in a directory. Typically used to generate a datasets for prior
    preservation / regularization.

    Args:
        out_dir (str): The output directory to create.
        model (str): The name or path of the diffusers pipeline to generate with.
        sd_version (PipelineVersionEnum): The model version.
        prompt (str): The prompt to generate images with.
        num_images (int): The number of images to generate.
        height (int): The output image height in pixels (recommended to match the resolution that the model was trained
            with).
        width (int): The output image width in pixels (recommended to match the resolution that the model was trained
            with).
        seed (int, optional): A seed for repeatability. Defaults to 0.
        torch_dtype (torch.dtype, optional): The torch dtype. Defaults to torch.float16.
        torch_device (str, optional): The torch device. Defaults to "cuda".
        enable_cpu_offload (bool, optional): If True, models will be loaded onto the GPU one by one to conserve VRAM.
            Defaults to False.
    """

    pipeline = load_pipeline(model, pipeline_version)

    pipeline.to(torch_dtype=torch_dtype)
    if enable_cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(torch_device=torch_device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=torch_device)
    if seed is not None:
        generator = generator.manual_seed(seed)

    os.makedirs(out_dir)

    with torch.no_grad():
        for i in tqdm(range(num_images)):
            image = pipeline(
                prompt,
                num_inference_steps=30,
                generator=generator,
                height=height,
                width=width,
            ).images[0]

            image.save(os.path.join(out_dir, f"{i:0>4}.jpg"))

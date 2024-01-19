import os
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from invoke_training.training._shared.data.datasets.image_pair_preference_dataset import ImagePairPreferenceDataset
from invoke_training.training._shared.stable_diffusion.model_loading_utils import load_pipeline_sd, load_pipeline_sdxl


class PipelineVersionEnum(Enum):
    SD = "SD"
    SDXL = "SDXL"


def generate_images(
    out_dir: str,
    model: str,
    hf_variant: str | None,
    pipeline_version: PipelineVersionEnum,
    prompts: list[str],
    set_size: int,
    num_sets: int,
    height: int,
    width: int,
    loras: Optional[list[tuple[Path, float]]] = None,
    ti_embeddings: Optional[list[str]] = None,
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
        pipeline_version (PipelineVersionEnum): The model version.
        prompt (str): The prompt to generate images with.
        set_size (int): The number of images in a 'set' for a given prompt.
        num_sets (int): The number of 'sets' to generate for each prompt.
        height (int): The output image height in pixels (recommended to match the resolution that the model was trained
            with).
        width (int): The output image width in pixels (recommended to match the resolution that the model was trained
            with).
        loras (list[tuple[Path, float]], optional): Paths to LoRA models to apply to the base model with associated
            weights.
        ti_embeddings (list[str], optional): Paths to TI embeddings to apply to the base model.
        seed (int, optional): A seed for repeatability. Defaults to 0.
        torch_dtype (torch.dtype, optional): The torch dtype. Defaults to torch.float16.
        torch_device (str, optional): The torch device. Defaults to "cuda".
        enable_cpu_offload (bool, optional): If True, models will be loaded onto the GPU one by one to conserve VRAM.
            Defaults to False.
    """

    if pipeline_version == PipelineVersionEnum.SD:
        pipeline = load_pipeline_sd(model_name_or_path=model, variant=hf_variant)
    elif pipeline_version == PipelineVersionEnum.SDXL:
        pipeline = load_pipeline_sdxl(model_name_or_path=model, variant=hf_variant)
    else:
        raise ValueError(f"Invalid pipeline version: {pipeline_version}.")

    loras = loras or []
    for lora in loras:
        lora_path, lora_scale = lora
        pipeline.load_lora_weights(str(lora_path), weight_name=str(lora_path.name))
        pipeline.fuse_lora(lora_scale=lora_scale)

    ti_embeddings = ti_embeddings or []
    for ti_embedding in ti_embeddings:
        pipeline.load_textual_inversion(ti_embedding)

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

    metadata = []

    total_images = num_sets * len(prompts) * set_size
    with torch.no_grad(), tqdm(total=total_images) as pbar:
        for prompt_idx in range(len(prompts)):
            for set_idx in range(num_sets):
                set_dir = os.path.join(out_dir, f"prompt-{prompt_idx:0>4}", f"set-{set_idx:0>4}")
                os.makedirs(set_dir)
                set_metadata_dict = {"prompt": prompts[prompt_idx]}
                for image_idx in range(set_size):
                    image = pipeline(
                        prompts[prompt_idx],
                        num_inference_steps=30,
                        generator=generator,
                        height=height,
                        width=width,
                    ).images[0]

                    image_path = os.path.join(set_dir, f"image-{image_idx}.jpg")
                    image.save(image_path)
                    set_metadata_dict[f"image_{image_idx}"] = os.path.relpath(image_path, start=out_dir)
                    set_metadata_dict[f"prefer_{image_idx}"] = False
                    pbar.update(1)
                metadata.append(set_metadata_dict)

    ImagePairPreferenceDataset.save_metadata(metadata=metadata, dataset_dir=out_dir)

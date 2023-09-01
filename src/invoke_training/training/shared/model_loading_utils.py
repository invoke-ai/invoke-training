import os
import typing
from enum import Enum

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline


class PipelineVersionEnum(Enum):
    SD = "SD"
    SDXL = "SDXL"


def load_pipeline(
    model_name_or_path: str, pipeline_version: PipelineVersionEnum
) -> typing.Union[StableDiffusionPipeline, StableDiffusionXLPipeline]:
    """Load a Stable Diffusion pipeline from disk.

    Args:
        model_name_or_path (str): The name or path of the pipeline to load. Can be in diffusers format, or a single
            stable diffusion checkpoint file. (E.g. 'runwayml/stable-diffusion-v1-5',
            'stabilityai/stable-diffusion-xl-base-1.0', '/path/to/realisticVisionV51_v51VAE.safetensors', etc. )
        pipeline_version (PipelineVersionEnum): The pipeline version.

    Returns:
        typing.Union[StableDiffusionPipeline, StableDiffusionXLPipeline]: The loaded pipeline.
    """
    if pipeline_version == PipelineVersionEnum.SD:
        pipeline_class = StableDiffusionPipeline
    elif pipeline_version == PipelineVersionEnum.SDXL:
        pipeline_class = StableDiffusionXLPipeline
    else:
        raise ValueError(f"Unsupported pipeline_version: '{pipeline_version}'.")

    if os.path.isfile(model_name_or_path):
        return pipeline_class.from_single_file(model_name_or_path, load_safety_checker=False)

    return pipeline_class.from_pretrained(
        model_name_or_path,
        safety_checker=None,
        requires_safety_checker=False,
    )

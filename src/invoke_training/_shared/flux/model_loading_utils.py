import logging
import os
import torch
from enum import Enum

import torch
from diffusers import (
    AutoencoderKL,
    FluxPipeline,
    FluxTransformer2DModel,
    FlowMatchEulerDiscreteScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from invoke_training._shared.checkpoints.serialization import load_state_dict


class PipelineVersionEnum(Enum):
    FLUX = "FLUX"

def load_pipeline(
    logger: logging.Logger,
    model_name_or_path: str = "black-forest-labs/FLUX.1-dev",
    pipeline_version: PipelineVersionEnum = PipelineVersionEnum.FLUX,
    transformer_path: str | None = None,
    text_encoder_1_path: str | None = None,
    text_encoder_2_path: str | None = None,
    torch_dtype: torch.dtype | None = None,
) -> FluxPipeline:
    """Load a Flux pipeline with optional custom components from .safetensors files.
    
    Args:
        logger: Logger instance
        model_name_or_path: Base model path or repository
        pipeline_version: Pipeline version (currently only FLUX supported)
        transformer_path: Path to custom transformer .safetensors file
        text_encoder_1_path: Path to custom CLIP text encoder .safetensors file
        text_encoder_2_path: Path to custom T5 text encoder .safetensors file
        torch_dtype: Desired dtype for the models
        
    Returns:
        FluxPipeline: Configured pipeline with custom components if specified
    """
    if pipeline_version != PipelineVersionEnum.FLUX:
        raise ValueError(f"Invalid pipeline version: {pipeline_version}")

    # Prepare kwargs for from_pretrained
    kwargs = {
        "torch_dtype": torch_dtype
    }

    # Add components only if custom paths are provided
    if transformer_path is not None:
        # load_model_from_file_or_pretrained(FluxTransformer2DModel, transformer_path, torch_dtype=torch_dtype, use_safetensors=True, subfolder="transformer")
        kwargs["transformer"] = FluxTransformer2DModel.from_pretrained(
            transformer_path,
            torch_dtype=torch_dtype,
        )
        logger.info(f"Loading custom transformer from {transformer_path}")


    if text_encoder_1_path is not None:
        logger.info(f"Loading custom CLIP text encoder from {text_encoder_1_path}")
        kwargs["text_encoder"] = CLIPTextModel.from_pretrained(
            text_encoder_1_path,
            torch_dtype=torch_dtype
        )

    if text_encoder_2_path is not None:
        logger.info(f"Loading custom T5 text encoder from {text_encoder_2_path}")
        kwargs["text_encoder_2"] = T5EncoderModel.from_pretrained(
            text_encoder_2_path,
            torch_dtype=torch_dtype
        )

    # Load the pipeline with any custom components
    pipeline = FluxPipeline.from_pretrained(model_name_or_path, **kwargs)

    return pipeline


def load_models_flux(
    logger: logging.Logger,
    model_name_or_path: str = "black-forest-labs/FLUX.1-dev",
    dtype: torch.dtype | None = None,
    transformer_path: str | None = None,
    text_encoder_1_path: str | None = None,
    text_encoder_2_path: str | None = None,
) -> tuple[CLIPTokenizer, FlowMatchEulerDiscreteScheduler, CLIPTextModel, AutoencoderKL, FluxTransformer2DModel]:
    """Load all models required for training from disk, transfer them to the
    target training device and cast their weight dtypes.

    Args:
        logger: Logger instance
        model_name_or_path: Base model path or repository
        dtype: Desired dtype for the models
        transformer_path: Path to custom transformer .safetensors file
        text_encoder_1_path: Path to custom CLIP text encoder .safetensors file
        text_encoder_2_path: Path to custom T5 text encoder .safetensors file
    """

    pipeline: FluxPipeline = load_pipeline(
        logger=logger,
        model_name_or_path=model_name_or_path,
        pipeline_version=PipelineVersionEnum.FLUX,
        transformer_path=transformer_path,
        text_encoder_1_path=text_encoder_1_path,
        text_encoder_2_path=text_encoder_2_path,
        torch_dtype=dtype,
    )


    # Tokenizers and text encoders.
    tokenizer_1: CLIPTokenizer = pipeline.tokenizer
    text_encoder_1: CLIPTextModel = pipeline.text_encoder

    tokenizer_2: T5Tokenizer = pipeline.tokenizer_2
    text_encoder_2: T5EncoderModel = pipeline.text_encoder_2

    # Transformer and Scheduler
    transformer: FluxTransformer2DModel = pipeline.transformer
    noise_scheduler: FlowMatchEulerDiscreteScheduler = pipeline.scheduler

    # Decoder 
    vae: AutoencoderKL = pipeline.vae

    # Log component status
    logger.info(f"Pipeline components loaded: tokenizer_1={tokenizer_1 is not None}, "
                f"text_encoder_1={text_encoder_1 is not None}, "
                f"tokenizer_2={tokenizer_2 is not None}, "
                f"text_encoder_2={text_encoder_2 is not None}, "
                f"transformer={transformer is not None}, "
                f"vae={vae is not None}")

    # Check for None components
    if text_encoder_1 is None:
        raise ValueError("text_encoder_1 failed to load. Check if you have access to the model repository and are properly authenticated.")
    if text_encoder_2 is None:
        raise ValueError("text_encoder_2 failed to load. Check if you have access to the model repository and are properly authenticated.")
    if transformer is None:
        raise ValueError("transformer failed to load. Check if you have access to the model repository and are properly authenticated.")
    if vae is None:
        raise ValueError("vae failed to load. Check if you have access to the model repository and are properly authenticated.")

    # Disable gradient calculation for model weights to save memory.
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    if dtype is not None:
        text_encoder_1 = text_encoder_1.to(dtype=dtype)
        text_encoder_2 = text_encoder_2.to(dtype=dtype)
        vae = vae.to(dtype=dtype)
        transformer = transformer.to(dtype=dtype)

    # Put models in 'eval' mode.
    text_encoder_1.eval()
    text_encoder_2.eval()
    vae.eval()
    transformer.eval()

    return tokenizer_1, tokenizer_2, noise_scheduler, text_encoder_1, text_encoder_2, vae, transformer


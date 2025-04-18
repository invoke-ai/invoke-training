import logging
import os
import typing
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
) -> FluxPipeline:
    if pipeline_version == PipelineVersionEnum.FLUX:
        pipeline = FluxPipeline.from_pretrained(model_name_or_path)
    else:
        raise ValueError(f"Invalid pipeline version: {pipeline_version}")
    return pipeline


def load_models_flux(
    logger: logging.Logger,
    model_name_or_path: str = "black-forest-labs/FLUX.1-dev",
    base_embeddings: dict[str, str] = None,
    dtype: torch.dtype | None = None,
) -> tuple[CLIPTokenizer, FlowMatchEulerDiscreteScheduler, CLIPTextModel, AutoencoderKL, FluxTransformer2DModel]:
    """Load all models required for training from disk, transfer them to the
    target training device and cast their weight dtypes.
    """
    base_embeddings = base_embeddings or {}

    pipeline: FluxPipeline = load_pipeline(
        logger=logger,
        model_name_or_path=model_name_or_path,
        pipeline_version=PipelineVersionEnum.FLUX,
    )

    for token, embedding_path in base_embeddings.items():
        pipeline.load_textual_inversion(embedding_path, token=token)

    # Tokenizers and text encoders.
    tokenizer_1: CLIPTokenizer = pipeline.tokenizer
    text_encoder_1: CLIPTextModel = pipeline.text_encoder

    tokenizer_2: T5Tokenizer = pipeline.tokenizer_2
    text_encoder_2: T5EncoderModel = pipeline.text_encoder_2


    # Diffuser and Scheduler
    transformer: FluxTransformer2DModel = pipeline.transformer
    noise_scheduler: FlowMatchEulerDiscreteScheduler = pipeline.scheduler

    # Decoder 
    vae: AutoencoderKL = pipeline.vae

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


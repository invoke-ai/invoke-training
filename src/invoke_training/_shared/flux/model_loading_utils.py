import logging
import os
import typing
from enum import Enum

import torch
from diffusers import (
    AutoencoderKL,
    FluxPipeline,
    FluxTransformer2DModel,
)
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from invoke_training._shared.checkpoints.serialization import load_state_dict


class PipelineVersionEnum(Enum):
    SD = "SD"
    SDXL = "SDXL"


def load_models_flux(
    logger: logging.Logger,
    model_name_or_path: str = "black-forest-labs/FLUX.1-dev",
    base_embeddings: dict[str, str] = None,
    dtype: torch.dtype | None = None,
) -> tuple[CLIPTokenizer, DDPMScheduler, CLIPTextModel, AutoencoderKL, UNet2DConditionModel]:
    """Load all models required for training from disk, transfer them to the
    target training device and cast their weight dtypes.
    """
    base_embeddings = base_embeddings or {}

    pipeline: StableDiffusionPipeline = load_pipeline(
        logger=logger,
        model_name_or_path=model_name_or_path,
    )

    for token, embedding_path in base_embeddings.items():
        pipeline.load_textual_inversion(embedding_path, token=token)

    # Tokenizers and text encoders.
    tokenizer_1: CLIPTokenizer = pipeline.tokenizer
    text_encoder_1: CLIPTextModel = pipeline.text_encoder

    tokenizer_2: T5Tokenizer = pipeline.tokenizer_2
    text_encoder_2: T5EncoderModel = pipeline.text_encoder_2


    # Diffuser and Scheduler
    diffuser: FluxTransformer2DModel = pipeline.unet
    noise_scheduler: FlowMatchEulerDiscreteScheduler = pipeline.scheduler

    # Decoder 
    vae: AutoencoderKL = pipeline.vae

    # Disable gradient calculation for model weights to save memory.
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    diffuser.requires_grad_(False)

    if dtype is not None:
        text_encoder = text_encoder.to(dtype=dtype)
        vae = vae.to(dtype=dtype)
        unet = unet.to(dtype=dtype)

    # Put models in 'eval' mode.
    text_encoder.eval()
    vae.eval()
    unet.eval()

    return tokenizer_1, tokenizer_2, noise_scheduler, text_encoder_1, text_encoder_2, vae, diffuser


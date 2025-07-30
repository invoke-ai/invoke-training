import logging
import os
import typing
from enum import Enum

import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from invoke_training._shared.checkpoints.serialization import load_state_dict

HF_VARIANT_FALLBACKS = [None, "fp16"]


class PipelineVersionEnum(Enum):
    SD = "SD"
    SDXL = "SDXL"


def load_pipeline(
    logger: logging.Logger,
    model_name_or_path: str,
    pipeline_version: PipelineVersionEnum,
    torch_dtype: torch.dtype = None,
    variant: str | None = None,
) -> typing.Union[StableDiffusionPipeline, StableDiffusionXLPipeline]:
    """Load a Stable Diffusion pipeline from disk.

    Args:
        model_name_or_path (str): The name or path of the pipeline to load. Can be in diffusers format, or a single
            stable diffusion checkpoint file. (E.g. 'runwayml/stable-diffusion-v1-5',
            'stabilityai/stable-diffusion-xl-base-1.0', '/path/to/realisticVisionV51_v51VAE.safetensors', etc. )
        pipeline_version (PipelineVersionEnum): The pipeline version.
        variant (str | None): The Hugging Face Hub variant. Only applies if `model_name_or_path` is a HF Hub model name.

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
        return pipeline_class.from_single_file(
            model_name_or_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            feature_extractor=None,
        )

    return from_pretrained_with_variant_fallback(
        logger=logger,
        model_class=pipeline_class,
        model_name_or_path=model_name_or_path,
        torch_dtype=torch_dtype,
        variant=variant,
        # kwargs
        safety_checker=None,
        requires_safety_checker=False,
    )


ModelT = typing.TypeVar("ModelT")


def from_pretrained_with_variant_fallback(
    logger: logging.Logger,
    model_class: typing.Type[ModelT],
    model_name_or_path: str,
    torch_dtype: torch.dtype | None = None,
    variant: str | None = None,
    **kwargs,
) -> ModelT:
    """A wrapper for .from_pretrained() that tries multiple variants if the initial one fails."""
    variants_to_try = [variant] + [v for v in HF_VARIANT_FALLBACKS if v != variant]

    model: ModelT | None = None
    for variant_to_try in variants_to_try:
        if variant_to_try != variant:
            logger.warning(f"Trying fallback variant '{variant_to_try}'.")
        try:
            model = model_class.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                variant=variant_to_try,
                **kwargs,
            )
        except (OSError, ValueError) as e:
            error_str = str(e)
            if "no file named" in error_str or "no such modeling files are available" in error_str:
                # Ok; we'll try the variant fallbacks.
                logger.warning(f"Failed to load '{model_name_or_path}' with variant '{variant_to_try}'. Error: {e}.")
            else:
                raise

        if model is not None:
            break

    if model is None:
        raise RuntimeError(f"Failed to load model '{model_name_or_path}'.")
    return model


def load_models_sd(
    logger: logging.Logger,
    model_name_or_path: str,
    hf_variant: str | None = None,
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
        pipeline_version=PipelineVersionEnum.SD,
        variant=hf_variant,
    )

    for token, embedding_path in base_embeddings.items():
        pipeline.load_textual_inversion(embedding_path, token=token)

    # Extract sub-models from the pipeline.
    tokenizer: CLIPTokenizer = pipeline.tokenizer
    text_encoder: CLIPTextModel = pipeline.text_encoder
    vae: AutoencoderKL = pipeline.vae
    unet: UNet2DConditionModel = pipeline.unet
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
        steps_offset=1,
    )

    # Disable gradient calculation for model weights to save memory.
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    if dtype is not None:
        text_encoder = text_encoder.to(dtype=dtype)
        vae = vae.to(dtype=dtype)
        unet = unet.to(dtype=dtype)

    # Put models in 'eval' mode.
    text_encoder.eval()
    vae.eval()
    unet.eval()

    return tokenizer, noise_scheduler, text_encoder, vae, unet


def load_models_sdxl(
    logger: logging.Logger,
    model_name_or_path: str,
    hf_variant: str | None = None,
    vae_model: str | None = None,
    base_embeddings: dict[str, str] = None,
    dtype: torch.dtype | None = None,
) -> tuple[
    CLIPTokenizer,
    CLIPTokenizer,
    DDPMScheduler,
    CLIPTextModel,
    CLIPTextModel,
    AutoencoderKL,
    UNet2DConditionModel,
]:
    """Load all models required for training, transfer them to the target training device and cast their weight
    dtypes.
    """
    base_embeddings = base_embeddings or {}

    pipeline: StableDiffusionXLPipeline = load_pipeline(
        logger=logger,
        model_name_or_path=model_name_or_path,
        pipeline_version=PipelineVersionEnum.SDXL,
        variant=hf_variant,
    )

    for token, embedding_path in base_embeddings.items():
        state_dict = load_state_dict(embedding_path)
        pipeline.load_textual_inversion(
            state_dict["clip_l"],
            token=token,
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
        )
        pipeline.load_textual_inversion(
            state_dict["clip_g"],
            token=token,
            text_encoder=pipeline.text_encoder_2,
            tokenizer=pipeline.tokenizer_2,
        )

    # Extract sub-models from the pipeline.
    tokenizer_1: CLIPTokenizer = pipeline.tokenizer
    tokenizer_2: CLIPTokenizer = pipeline.tokenizer_2
    text_encoder_1: CLIPTextModel = pipeline.text_encoder
    text_encoder_2: CLIPTextModel = pipeline.text_encoder_2
    vae: AutoencoderKL = pipeline.vae
    unet: UNet2DConditionModel = pipeline.unet
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
        steps_offset=1,
    )

    if vae_model is not None:
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(vae_model)

    # Disable gradient calculation for model weights to save memory.
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    if dtype is not None:
        text_encoder_1 = text_encoder_1.to(dtype=dtype)
        text_encoder_2 = text_encoder_2.to(dtype=dtype)
        vae = vae.to(dtype=dtype)
        unet = unet.to(dtype=dtype)

    # Put models in 'eval' mode.
    text_encoder_1.eval()
    text_encoder_2.eval()
    vae.eval()
    unet.eval()

    return tokenizer_1, tokenizer_2, noise_scheduler, text_encoder_1, text_encoder_2, vae, unet

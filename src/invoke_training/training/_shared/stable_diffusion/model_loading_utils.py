import os

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer


def load_pipeline_sd(model_name_or_path: str, variant: str | None = None) -> StableDiffusionPipeline:
    """Load a Stable Diffusion pipeline from disk."""
    if os.path.isfile(model_name_or_path):
        return StableDiffusionPipeline.from_single_file(model_name_or_path, load_safety_checker=False)

    return StableDiffusionPipeline.from_pretrained(
        model_name_or_path,
        safety_checker=None,
        variant=variant,
        requires_safety_checker=False,
    )


def load_pipeline_sdxl(model_name_or_path: str, variant: str | None = None) -> StableDiffusionXLPipeline:
    if os.path.isfile(model_name_or_path):
        return StableDiffusionXLPipeline.from_single_file(model_name_or_path, load_safety_checker=False)

    return StableDiffusionXLPipeline.from_pretrained(
        model_name_or_path,
        safety_checker=None,
        variant=variant,
        requires_safety_checker=False,
    )


def load_models_sd(
    model_name_or_path: str, hf_variant: str | None = None
) -> tuple[CLIPTokenizer, DDPMScheduler, CLIPTextModel, AutoencoderKL, UNet2DConditionModel]:
    """Load all models required for training from disk, transfer them to the
    target training device and cast their weight dtypes.

    Returns:
        tuple[
            CLIPTokenizer,
            DDPMScheduler,
            CLIPTextModel,
            AutoencoderKL,
            UNet2DConditionModel,
        ]: A tuple of loaded models.
    """
    pipeline: StableDiffusionPipeline = load_pipeline_sd(model_name_or_path=model_name_or_path, variant=hf_variant)

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

    # Put models in 'eval' mode.
    text_encoder.eval()
    vae.eval()
    unet.eval()

    return tokenizer, noise_scheduler, text_encoder, vae, unet


def load_models_sdxl(
    model_name_or_path: str, hf_variant: str | None = None, vae_model: str | None = None
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
    pipeline: StableDiffusionXLPipeline = load_pipeline_sdxl(model_name_or_path=model_name_or_path, variant=hf_variant)

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

    # Put models in 'eval' mode.
    text_encoder_1.eval()
    text_encoder_2.eval()
    vae.eval()
    unet.eval()

    return tokenizer_1, tokenizer_2, noise_scheduler, text_encoder_1, text_encoder_2, vae, unet

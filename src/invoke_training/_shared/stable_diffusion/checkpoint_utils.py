from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def save_sdxl_diffusers_unet_checkpoint(checkpoint_path: Path | str, unet: UNet2DConditionModel):
    unet.save_pretrained(Path(checkpoint_path) / "unet")


def save_sdxl_diffusers_checkpoint(
    checkpoint_path: Path | str,
    vae: AutoencoderKL,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModel,
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    noise_scheduler: DDPMScheduler,
    unet: UNet2DConditionModel,
    save_dtype: torch.dtype,
):
    # Record original device and dtype so that we can restore it afterward.
    # TODO(ryand): This method of restoring original device/dtype is a bit naive. It does not handle mixed precisions
    # within a model, and results in a loss of precision if the save_dtype is lower precision than the model dtype. We
    # may need to revisit this.
    model_list = [vae, text_encoder_1, text_encoder_2, unet]
    original_devices = [model.device for model in model_list]
    original_dtypes = [model.dtype for model in model_list]

    # Create pipeline.
    pipeline = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=noise_scheduler,
    )
    pipeline = pipeline.to(device="cpu", dtype=save_dtype)

    # Save pipeline.
    pipeline.save_pretrained(checkpoint_path)

    # Restore original device/dtype.
    for model, device, dtype in zip(model_list, original_devices, original_dtypes, strict=True):
        model = model.to(device=device, dtype=dtype)

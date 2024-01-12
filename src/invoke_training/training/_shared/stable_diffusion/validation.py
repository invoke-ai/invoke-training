import logging
import os

import numpy as np
import torch
import torch.utils.data
from accelerate import Accelerator
from accelerate.hooks import remove_hook_from_module
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from invoke_training.config.pipelines.finetune_lora_config import FinetuneLoRASDConfig, FinetuneLoRASDXLConfig


def generate_validation_images_sd(
    epoch: int,
    out_dir: str,
    accelerator: Accelerator,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    noise_scheduler: DDPMScheduler,
    unet: UNet2DConditionModel,
    config: FinetuneLoRASDConfig,
    logger: logging.Logger,
):
    """Generate validation images for the purpose of tracking image generation behaviour on fixed prompts throughout
    training.

    Args:
        epoch (int): Epoch number, for reporting purposes.
        out_dir (str): The output directory where the validation images will be stored.
        accelerator (Accelerator): Accelerator
        vae (AutoencoderKL):
        text_encoder (CLIPTextModel):
        tokenizer (CLIPTokenizer):
        noise_scheduler (DDPMScheduler):
        unet (UNet2DConditionModel):
        config (FinetuneLoRASDConfig): Training configs.
        logger (logging.Logger): Logger.
    """
    logger.info("Generating validation images.")

    # Record original model devices so that we can restore this state after running the pipeline with CPU model
    # offloading.
    unet_device = unet.device
    vae_device = vae.device
    text_encoder_device = text_encoder.device

    # Create pipeline.
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
        # TODO(ryand): Add safety checker support.
        requires_safety_checker=False,
    )
    if config.enable_cpu_offload_during_validation:
        pipeline.enable_model_cpu_offload(accelerator.device.index or 0)
    else:
        pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # Run inference.
    with torch.no_grad():
        for prompt_idx, prompt in enumerate(config.validation_prompts):
            generator = torch.Generator(device=accelerator.device)
            if config.seed is not None:
                generator = generator.manual_seed(config.seed)

            images = []
            for _ in range(config.num_validation_images_per_prompt):
                with accelerator.autocast():
                    images.append(
                        pipeline(
                            prompt,
                            num_inference_steps=30,
                            generator=generator,
                            height=config.data_loader.image_transforms.resolution,
                            width=config.data_loader.image_transforms.resolution,
                        ).images[0]
                    )

            # Save images to disk.
            validation_dir = os.path.join(
                out_dir,
                "validation",
                f"epoch_{epoch:0>8}",
                f"prompt_{prompt_idx:0>4}",
            )
            os.makedirs(validation_dir)
            for image_idx, image in enumerate(images):
                image.save(os.path.join(validation_dir, f"{image_idx:0>4}.jpg"))

            # Log images to trackers. Currently, only tensorboard is supported.
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(
                        f"validation (prompt {prompt_idx})",
                        np_images,
                        epoch,
                        dataformats="NHWC",
                    )

    del pipeline
    torch.cuda.empty_cache()

    # Remove hooks from models.
    # HACK(ryand): Hooks get added when calling `pipeline.enable_model_cpu_offload(...)`, but `StableDiffusionPipeline`
    # does not offer a way to clean them up so we have to do this manually.
    for model in [unet, vae, text_encoder]:
        remove_hook_from_module(model)

    # Restore models to original devices.
    unet.to(unet_device)
    vae.to(vae_device)
    text_encoder.to(text_encoder_device)


def generate_validation_images_sdxl(
    epoch: int,
    out_dir: str,
    accelerator: Accelerator,
    vae: AutoencoderKL,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModel,
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    noise_scheduler: DDPMScheduler,
    unet: UNet2DConditionModel,
    config: FinetuneLoRASDXLConfig,
    logger: logging.Logger,
):
    """Generate validation images for the purpose of tracking image generation behaviour on fixed prompts throughout
    training.
    """
    logger.info("Generating validation images.")

    # Record original model devices so that we can restore this state after running the pipeline with CPU model
    # offloading.
    unet_device = unet.device
    vae_device = vae.device
    text_encoder_1_device = text_encoder_1.device
    text_encoder_2_device = text_encoder_2.device

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
    if config.enable_cpu_offload_during_validation:
        pipeline.enable_model_cpu_offload(accelerator.device.index or 0)
    else:
        pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # Run inference.
    with torch.no_grad():
        for prompt_idx, prompt in enumerate(config.validation_prompts):
            generator = torch.Generator(device=accelerator.device)
            if config.seed is not None:
                generator = generator.manual_seed(config.seed)

            images = []
            for _ in range(config.num_validation_images_per_prompt):
                with accelerator.autocast():
                    images.append(
                        pipeline(
                            prompt,
                            num_inference_steps=30,
                            generator=generator,
                            height=config.data_loader.image_transforms.resolution,
                            width=config.data_loader.image_transforms.resolution,
                        ).images[0]
                    )

            # Save images to disk.
            validation_dir = os.path.join(
                out_dir,
                "validation",
                f"epoch_{epoch:0>8}",
                f"prompt_{prompt_idx:0>4}",
            )
            os.makedirs(validation_dir)
            for image_idx, image in enumerate(images):
                image.save(os.path.join(validation_dir, f"{image_idx:0>4}.jpg"))

            # Log images to trackers. Currently, only tensorboard is supported.
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(
                        f"validation (prompt {prompt_idx})",
                        np_images,
                        epoch,
                        dataformats="NHWC",
                    )

    del pipeline
    torch.cuda.empty_cache()

    # Remove hooks from models.
    # HACK(ryand): Hooks get added when calling `pipeline.enable_model_cpu_offload(...)`, but
    # `StableDiffusionXLPipeline` does not offer a way to clean them up so we have to do this manually.
    for model in [unet, vae, text_encoder_1, text_encoder_2]:
        remove_hook_from_module(model)

    # Restore models to original devices.
    unet.to(unet_device)
    vae.to(vae_device)
    text_encoder_1.to(text_encoder_1_device)
    text_encoder_2.to(text_encoder_2_device)

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

from invoke_training._shared.data.utils.resolution import Resolution
from invoke_training.pipelines.stable_diffusion.lora.config import SdLoraConfig
from invoke_training.pipelines.stable_diffusion_xl.lora.config import SdxlLoraConfig


def split_prompt(prompt: str) -> tuple[str, str]:
    """
    Extracts two strings from an input string:
    - The first string contains everything outside square brackets.
    - The second string contains everything inside the square brackets.
    - square brackets are also removed from the prompt
    - square brackets can be escaped and ignored with a backslash
    """
    positive_prompt = ""
    negative_prompt = ""
    brackets_depth = 0
    escaped = False

    for char in prompt or "":
        if char == "[" and not escaped:
            negative_prompt += " "
            brackets_depth += 1
        elif char == "]" and not escaped:
            brackets_depth -= 1
            char = " "
        elif brackets_depth > 0:
            negative_prompt += char
        else:
            positive_prompt += char
            brackets_depth = 0

        # keep track of the escape char but only if it isn't escaped already
        if char == "\\" and not escaped:
            escaped = True
        else:
            escaped = False

    return positive_prompt, negative_prompt


def generate_validation_images_sd(
    step: int,
    out_dir: str,
    accelerator: Accelerator,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    noise_scheduler: DDPMScheduler,
    unet: UNet2DConditionModel,
    config: SdLoraConfig,
    logger: logging.Logger,
    prefix="epoch",
):
    """Generate validation images for the purpose of tracking image generation behaviour on fixed prompts throughout
    training.
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

    validation_resolution = Resolution.parse(config.data_loader.resolution)

    # Run inference.
    with torch.no_grad():
        for prompt_idx, prompt in enumerate(config.validation_prompts):
            generator = torch.Generator(device=accelerator.device)
            if config.seed is not None:
                generator = generator.manual_seed(config.seed)

            positive_prompt, negative_prompt = split_prompt(prompt)
            logger.info(f"SD-prompt {prompt_idx}:{prompt}, pos:{positive_prompt}, neg:{negative_prompt}")

            images = []
            for _ in range(config.num_validation_images_per_prompt):
                with accelerator.autocast():
                    images.append(
                        pipeline(
                            positive_prompt,
                            num_inference_steps=30,
                            generator=generator,
                            height=validation_resolution.height,
                            width=validation_resolution.width,
                            negative_prompt=negative_prompt,
                        ).images[0]
                    )

            # Save images to disk.
            validation_dir = os.path.join(
                out_dir,
                "validation",
                f"{prefix}_{step:0>8}",
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
                        f"validation ({prefix}) (prompt {prompt_idx})",
                        np_images,
                        step,
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
    step: int,
    out_dir: str,
    accelerator: Accelerator,
    vae: AutoencoderKL,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModel,
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    noise_scheduler: DDPMScheduler,
    unet: UNet2DConditionModel,
    config: SdxlLoraConfig,
    logger: logging.Logger,
    prefix: str = "epoch",
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

    validation_resolution = Resolution.parse(config.data_loader.resolution)

    # Run inference.
    with torch.no_grad():
        for prompt_idx, prompt in enumerate(config.validation_prompts):
            generator = torch.Generator(device=accelerator.device)
            if config.seed is not None:
                generator = generator.manual_seed(config.seed)

            positive_prompt, negative_prompt = split_prompt(prompt)
            logger.info(f"SDXL-prompt {prompt_idx}:{prompt}, pos:{positive_prompt}, neg:{negative_prompt}")

            images = []
            for _ in range(config.num_validation_images_per_prompt):
                with accelerator.autocast():
                    images.append(
                        pipeline(
                            positive_prompt,
                            num_inference_steps=30,
                            generator=generator,
                            height=validation_resolution.height,
                            width=validation_resolution.width,
                            negative_prompt=negative_prompt,
                        ).images[0]
                    )

            # Save images to disk.
            validation_dir = os.path.join(
                out_dir,
                "validation",
                f"{prefix}_{step:0>8}",
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
                        f"validation ({prefix}) (prompt {prompt_idx})",
                        np_images,
                        step,
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

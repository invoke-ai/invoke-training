import logging
import os

import numpy as np
import torch
import torch.utils.data
from accelerate import Accelerator
from accelerate.hooks import remove_hook_from_module
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel

from invoke_training._shared.data.utils.resolution import Resolution
from invoke_training.pipelines.callbacks import PipelineCallbacks, ValidationImage, ValidationImages
from invoke_training.pipelines.flux.lora.config import FluxLoraConfig


def generate_validation_images_flux(  # noqa: C901
    epoch: int,
    step: int,
    out_dir: str,
    accelerator: Accelerator,
    vae: AutoencoderKL,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModel,
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    noise_scheduler: FlowMatchEulerDiscreteScheduler,
    transformer: FluxTransformer2DModel | PeftModel,
    config: FluxLoraConfig,
    logger: logging.Logger,
    callbacks: list[PipelineCallbacks] | None = None,
):
    """Generate validation images for the purpose of tracking image generation behaviour on fixed prompts throughout
    training.
    """
    # Record original model devices so that we can restore this state after running the pipeline with CPU model
    # offloading.
    transformer_device = transformer.device
    vae_device = vae.device
    text_encoder_1_device = text_encoder_1.device
    text_encoder_2_device = text_encoder_2.device

    # Create pipeline.
    pipeline = FluxPipeline(
        vae=vae,
        text_encoder=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        transformer=transformer,
        scheduler=noise_scheduler,
    )
    if config.enable_cpu_offload_during_validation:
        pipeline.enable_model_cpu_offload(accelerator.device.index or 0)
    else:
        pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    validation_resolution = Resolution.parse(config.data_loader.resolution)

    validation_images = ValidationImages(images=[], epoch=epoch, step=step)

    validation_step_dir = os.path.join(out_dir, "validation", f"epoch_{epoch:0>8}-step_{step:0>8}")
    logger.info(f"Generating validation images ({validation_step_dir}).")

    # Run inference.
    with torch.no_grad():
        for prompt_idx in range(len(config.validation_prompts)):
            positive_prompt = config.validation_prompts[prompt_idx]
            negative_prompt = None
            if config.negative_validation_prompts is not None:
                negative_prompt = config.negative_validation_prompts[prompt_idx]
            logger.info(f"Validation prompt {prompt_idx}, pos: '{positive_prompt}', neg: '{negative_prompt or ''}'")

            generator = torch.Generator(device=accelerator.device)
            if config.seed is not None:
                generator = generator.manual_seed(config.seed)

            images = []
            for _ in range(config.num_validation_images_per_prompt):
                with accelerator.autocast():
                    images.append(
                        pipeline(
                            positive_prompt,
                            num_inference_steps=20,
                            generator=generator,
                            height=validation_resolution.height,
                            width=validation_resolution.width,
                            negative_prompt=negative_prompt,
                        ).images[0]
                    )

            # Save images to disk.
            validation_prompt_dir = os.path.join(validation_step_dir, f"prompt_{prompt_idx:0>4}")
            os.makedirs(validation_prompt_dir)
            for image_idx, image in enumerate(images):
                image_path = os.path.join(validation_prompt_dir, f"{image_idx:0>4}.jpg")
                validation_images.images.append(
                    ValidationImage(file_path=image_path, prompt=positive_prompt, image_idx=image_idx)
                )
                image.save(image_path)

            # Log images to trackers. Currently, only tensorboard is supported.
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(
                        f"validation (prompt {prompt_idx})",
                        np_images,
                        step,
                        dataformats="NHWC",
                    )

    del pipeline
    torch.cuda.empty_cache()

    for model in [transformer, vae, text_encoder_1, text_encoder_2]:
        remove_hook_from_module(model)

    # Restore models to original devices.
    transformer.to(transformer_device)
    vae.to(vae_device)
    text_encoder_1.to(text_encoder_1_device)
    text_encoder_2.to(text_encoder_2_device)

    # Run callbacks.
    if callbacks is not None:
        for cb in callbacks:
            cb.on_save_validation_images(images=validation_images)

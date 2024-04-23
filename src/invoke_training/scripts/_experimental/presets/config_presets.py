from typing import Literal, Optional

from invoke_training.config.data.data_loader_config import (
    AspectRatioBucketConfig,
    ImageCaptionSDDataLoaderConfig,
    TextualInversionSDDataLoaderConfig,
)
from invoke_training.config.data.dataset_config import ImageCaptionJsonlDatasetConfig
from invoke_training.config.optimizer.optimizer_config import AdamOptimizerConfig
from invoke_training.pipelines.stable_diffusion.lora.config import SdLoraConfig
from invoke_training.pipelines.stable_diffusion.textual_inversion.config import SdTextualInversionConfig
from invoke_training.pipelines.stable_diffusion_xl.lora.config import SdxlLoraConfig
from invoke_training.pipelines.stable_diffusion_xl.textual_inversion.config import SdxlTextualInversionConfig
from invoke_training.scripts._experimental.presets.pipeline_config_override import PipelineConfigOverride
from invoke_training.scripts._experimental.presets.training_length_override import TrainingLengthOverride

# TODO(ryand): Increase this. It's only set to 1 for testing purposes.
# The maximum number of checkpoints to keep.
MAX_CHECKPOINTS = 1

NUM_VALIDATION_IMAGES_PER_PROMPT = 3

# Default aspect ratio bucket configs for SD and SDXL models respectively.
ASPECT_RATIO_BUCKET_CONFIG_SDXL = AspectRatioBucketConfig(
    target_resolution=1024,
    start_dim=512,
    end_dim=1536,
    divisible_by=128,
)
ASPECT_RATIO_BUCKET_CONFIG_SD = AspectRatioBucketConfig(
    target_resolution=512,
    start_dim=256,
    end_dim=768,
    divisible_by=64,
)

# Default resolution for SD and SDXL models respectively.
RESOLUTION_SDXL = 1024
RESOLUTION_SD = 512

# Caution: Increasing this value can significantly increase RAM requirements.
# See https://pytorch.org/docs/stable/data.html#multi-process-data-loading for more information.
DATALOADER_NUM_WORKERS = 2


def _should_keep_dataset_in_memory(dataset_size: int) -> bool:
    return dataset_size < 10


def get_sdxl_lora_preset_config(
    jsonl_path: str,
    dataset_size: int,
    model: str,
    vae_model: str | None,
    text_encoder_learning_rate: float,
    unet_learning_rate: float,
    caption_prefix: Optional[str],
    validation_prompts: list[str],
    overrides: list[PipelineConfigOverride],
) -> SdxlLoraConfig:
    """Prepare a configuration for training a general SDXL LoRA model."""

    config = SdxlLoraConfig(
        model=model,
        vae_model=vae_model,
        seed=0,
        base_output_dir="output",
        optimizer=AdamOptimizerConfig(),
        text_encoder_learning_rate=text_encoder_learning_rate,
        unet_learning_rate=unet_learning_rate,
        lr_scheduler="constant_with_warmup",
        lr_warmup_steps=200,
        lora_rank_dim=4,
        mixed_precision="fp16",
        gradient_checkpointing=True,
        max_checkpoints=MAX_CHECKPOINTS,
        validation_prompts=validation_prompts,
        num_validation_images_per_prompt=3,
        train_batch_size=4,
        data_loader=ImageCaptionSDDataLoaderConfig(
            dataset=ImageCaptionJsonlDatasetConfig(
                jsonl_path=jsonl_path,
                keep_in_memory=_should_keep_dataset_in_memory(dataset_size),
            ),
            resolution=RESOLUTION_SDXL,
            aspect_ratio_buckets=ASPECT_RATIO_BUCKET_CONFIG_SDXL,
            caption_prefix=caption_prefix,
            dataloader_num_workers=DATALOADER_NUM_WORKERS,
        ),
    )

    preset_overrides: list[PipelineConfigOverride] = [
        # Configure the training length and checkpoint frequency.
        TrainingLengthOverride(dataset_size),
    ]

    #  Note that we apply the caller-provided overrides before the preset overrides.
    for override in overrides + preset_overrides:
        override.apply_override(config)

    # TODO(ryand): Validate after all the modifications?
    return config


def get_sd_lora_preset_config(
    jsonl_path: str,
    dataset_size: int,
    model: str,
    text_encoder_learning_rate: float,
    unet_learning_rate: float,
    caption_prefix: Optional[str],
    validation_prompts: list[str],
    overrides: list[PipelineConfigOverride],
) -> SdLoraConfig:
    """Prepare a configuration for training a general SD1 LoRA model."""

    config = SdLoraConfig(
        model=model,
        seed=0,
        base_output_dir="output",
        optimizer=AdamOptimizerConfig(),
        text_encoder_learning_rate=text_encoder_learning_rate,
        unet_learning_rate=unet_learning_rate,
        lr_scheduler="constant_with_warmup",
        lr_warmup_steps=200,
        lora_rank_dim=4,
        mixed_precision="fp16",
        gradient_checkpointing=False,
        max_checkpoints=MAX_CHECKPOINTS,
        validation_prompts=validation_prompts,
        num_validation_images_per_prompt=3,
        train_batch_size=4,
        data_loader=ImageCaptionSDDataLoaderConfig(
            dataset=ImageCaptionJsonlDatasetConfig(
                jsonl_path=jsonl_path,
                keep_in_memory=_should_keep_dataset_in_memory(dataset_size),
            ),
            resolution=RESOLUTION_SD,
            aspect_ratio_buckets=ASPECT_RATIO_BUCKET_CONFIG_SD,
            caption_prefix=caption_prefix,
            dataloader_num_workers=DATALOADER_NUM_WORKERS,
        ),
    )

    preset_overrides: list[PipelineConfigOverride] = [
        # Configure the training length and checkpoint frequency.
        TrainingLengthOverride(dataset_size),
    ]

    #  Note that we apply the caller-provided overrides before the preset overrides.
    for override in overrides + preset_overrides:
        override.apply_override(config)

    # TODO(ryand): Validate after all the modifications?
    return config


def get_sdxl_ti_preset_config(
    jsonl_path: str,
    dataset_size: int,
    model: str,
    vae_model: str | None,
    placeholder_token: str,
    initializer_token: str,
    num_vectors: int,
    learning_rate: float,
    validation_prompts: list[str],
    caption_preset: Literal["style", "object"],
    overrides: list[PipelineConfigOverride],
) -> SdxlTextualInversionConfig:
    """Prepare a configuration for training a general SDXL TI model."""

    config = SdxlTextualInversionConfig(
        model=model,
        vae_model=vae_model,
        seed=0,
        base_output_dir="output",
        placeholder_token=placeholder_token,
        initializer_token=initializer_token,
        num_vectors=num_vectors,
        optimizer=AdamOptimizerConfig(learning_rate=learning_rate),
        lr_scheduler="constant_with_warmup",
        lr_warmup_steps=200,
        mixed_precision="fp16",
        max_checkpoints=MAX_CHECKPOINTS,
        gradient_checkpointing=True,
        validation_prompts=validation_prompts,
        num_validation_images_per_prompt=NUM_VALIDATION_IMAGES_PER_PROMPT,
        train_batch_size=4,
        data_loader=TextualInversionSDDataLoaderConfig(
            dataset=ImageCaptionJsonlDatasetConfig(
                jsonl_path=jsonl_path,
                keep_in_memory=_should_keep_dataset_in_memory(dataset_size),
            ),
            caption_preset=caption_preset,
            keep_original_captions=True,
            aspect_ratio_buckets=ASPECT_RATIO_BUCKET_CONFIG_SDXL,
            resolution=RESOLUTION_SDXL,
            dataloader_num_workers=DATALOADER_NUM_WORKERS,
        ),
    )

    preset_overrides: list[PipelineConfigOverride] = [
        # Configure the training length and checkpoint frequency.
        TrainingLengthOverride(dataset_size),
    ]

    #  Note that we apply the caller-provided overrides before the preset overrides.
    for override in overrides + preset_overrides:
        override.apply_override(config)

    # TODO(ryand): Validate after all the modifications?
    return config


def get_sd_ti_preset_config(
    jsonl_path: str,
    dataset_size: int,
    model: str,
    placeholder_token: str,
    initializer_token: str,
    num_vectors: int,
    learning_rate: float,
    validation_prompts: list[str],
    caption_preset: Literal["style", "object"],
    overrides: list[PipelineConfigOverride],
) -> SdTextualInversionConfig:
    """Prepare a configuration for training a general SDXL TI model."""

    config = SdTextualInversionConfig(
        model=model,
        seed=0,
        base_output_dir="output",
        placeholder_token=placeholder_token,
        initializer_token=initializer_token,
        num_vectors=num_vectors,
        optimizer=AdamOptimizerConfig(learning_rate=learning_rate),
        lr_scheduler="constant_with_warmup",
        lr_warmup_steps=200,
        mixed_precision="fp16",
        max_checkpoints=MAX_CHECKPOINTS,
        gradient_checkpointing=False,
        validation_prompts=validation_prompts,
        num_validation_images_per_prompt=NUM_VALIDATION_IMAGES_PER_PROMPT,
        train_batch_size=4,
        data_loader=TextualInversionSDDataLoaderConfig(
            dataset=ImageCaptionJsonlDatasetConfig(
                jsonl_path=jsonl_path,
                keep_in_memory=_should_keep_dataset_in_memory(dataset_size),
            ),
            caption_preset=caption_preset,
            keep_original_captions=True,
            aspect_ratio_buckets=ASPECT_RATIO_BUCKET_CONFIG_SD,
            resolution=RESOLUTION_SD,
            dataloader_num_workers=DATALOADER_NUM_WORKERS,
        ),
    )

    preset_overrides: list[PipelineConfigOverride] = [
        # Configure the training length and checkpoint frequency.
        TrainingLengthOverride(dataset_size),
    ]

    #  Note that we apply the caller-provided overrides before the preset overrides.
    for override in overrides + preset_overrides:
        override.apply_override(config)

    # TODO(ryand): Validate after all the modifications?
    return config

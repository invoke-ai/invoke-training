from pathlib import Path
from typing import Literal

import yaml
from pydantic import TypeAdapter

from invoke_training.config.data.data_loader_config import AspectRatioBucketConfig, TextualInversionSDDataLoaderConfig
from invoke_training.config.data.dataset_config import ImageCaptionJsonlDatasetConfig
from invoke_training.config.optimizer.optimizer_config import AdamOptimizerConfig
from invoke_training.config.pipeline_config import PipelineConfig
from invoke_training.pipelines.stable_diffusion_xl.lora.config import SdxlLoraConfig
from invoke_training.pipelines.stable_diffusion_xl.textual_inversion.config import SdxlTextualInversionConfig
from invoke_training.scripts._experimental.presets.pipeline_config_override import PipelineConfigOverride
from invoke_training.scripts._experimental.presets.training_length_override import TrainingLengthOverride

# TODO(ryand): Increase this. It's only set to 1 for testing purposes.
# The maximum number of checkpoints to keep.
MAX_CHECKPOINTS = 1

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


def _load_config_from_file(config_path: Path) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    pipeline_adapter: TypeAdapter[PipelineConfig] = TypeAdapter(PipelineConfig)
    train_config = pipeline_adapter.validate_python(cfg)
    return train_config


class JsonlPathOverride(PipelineConfigOverride):
    def __init__(self, jsonl_path: str):
        self._jsonl_path = jsonl_path

    def apply_override(self, config: PipelineConfig):
        config.data_loader.dataset.jsonl_path = self._jsonl_path


class BaseOutputDirOverride(PipelineConfigOverride):
    def __init__(self, base_output_dir: str):
        self._base_output_dir = base_output_dir

    def apply_override(self, config: PipelineConfig):
        config.base_output_dir = self._base_output_dir


class TrainBatchSizeOverride(PipelineConfigOverride):
    def __init__(self, train_batch_size: int):
        self._train_batch_size = train_batch_size

    def apply_override(self, config: PipelineConfig):
        config.train_batch_size = self._train_batch_size


class ValidationPromptsOverride(PipelineConfigOverride):
    def __init__(self, validation_prompts: list[str]):
        self._validation_prompts = validation_prompts

    def apply_override(self, config: PipelineConfig):
        config.validation_prompts = self._validation_prompts


def _prepare_config(config_path: Path, overrides: list[PipelineConfigOverride]) -> PipelineConfig:
    config = _load_config_from_file(config_path)
    for override in overrides:
        override.apply_override(config)
    return config


def get_sdxl_lora_preset_config(
    jsonl_path: str, dataset_size: int, validation_prompts: list[str], overrides: list[PipelineConfigOverride]
) -> SdxlLoraConfig:
    """Prepare a configuration for training a general SDXL LoRA model."""
    config_path = Path(__file__).parent / "configs/presets/sdxl_lora_preset_1x24gb.yaml"

    preset_overrides: list[PipelineConfigOverride] = [
        # Override the dataset path.
        JsonlPathOverride(jsonl_path),
        # Override the validation prompts.
        ValidationPromptsOverride(validation_prompts),
        # Configure the training length and checkpoint frequency.
        TrainingLengthOverride(dataset_size),
    ]

    #  Note that we apply the caller-provided overrides before the preset overrides.
    return _prepare_config(config_path, overrides + preset_overrides)


def get_sd_lora_preset_config(
    jsonl_path: str, dataset_size: int, validation_prompts: list[str], overrides: list[PipelineConfigOverride]
) -> SdxlLoraConfig:
    """Prepare a configuration for training a general SD1 LoRA model."""
    config_path = Path(__file__).parent / "configs/presets/sd_lora_preset_1x24gb.yaml"

    preset_overrides: list[PipelineConfigOverride] = [
        # Override the dataset path.
        JsonlPathOverride(jsonl_path),
        # Override the validation prompts.
        ValidationPromptsOverride(validation_prompts),
        # Configure the training length and checkpoint frequency.
        TrainingLengthOverride(dataset_size),
    ]

    # Note that we apply the caller-provided overrides before the preset overrides.
    return _prepare_config(config_path, overrides + preset_overrides)


def get_sdxl_ti_preset_config(
    jsonl_path: str,
    dataset_size: int,
    model: str,
    vae_model: str | None,
    placeholder_token: str,
    initializer_token: str,
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
        num_vectors=4,
        optimizer=AdamOptimizerConfig(learning_rate=learning_rate),
        lr_scheduler="constant_with_warmup",
        lr_warmup_steps=200,
        mixed_precision="fp16",
        max_checkpoints=MAX_CHECKPOINTS,
        gradient_checkpointing=True,
        validation_prompts=validation_prompts,
        num_validation_images_per_prompt=3,
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
    jsonl_path: str, dataset_size: int, validation_prompts: list[str], overrides: list[PipelineConfigOverride]
) -> SdxlLoraConfig:
    """Prepare a configuration for training a general SDXL TI model."""
    config_path = Path(__file__).parent / "configs/presets/sd_ti_preset_1x24gb.yaml"

    preset_overrides: list[PipelineConfigOverride] = [
        # Override the dataset path.
        JsonlPathOverride(jsonl_path),
        # Override the validation prompts.
        ValidationPromptsOverride(validation_prompts),
        # Configure the training length and checkpoint frequency.
        TrainingLengthOverride(dataset_size),
    ]

    #  Note that we apply the caller-provided overrides before the preset overrides.
    return _prepare_config(config_path, overrides + preset_overrides)

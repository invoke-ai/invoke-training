import math
from abc import ABC, abstractmethod
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


class PipelineConfigOverride(ABC):
    @abstractmethod
    def apply_override(self, config: PipelineConfig):
        pass


def _load_config_from_file(config_path: Path) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    pipeline_adapter: TypeAdapter[PipelineConfig] = TypeAdapter(PipelineConfig)
    train_config = pipeline_adapter.validate_python(cfg)
    return train_config


class TrainingLengthOverride(PipelineConfigOverride):
    """An override to configure the training length and checkpoint frequency.

    This override applies some simple heuristics based on the dataset size to obtain reasonable settings.
    """

    # TODO(ryand): Should there be a max_epochs limit?
    def __init__(
        self,
        dataset_size: int,
        target_steps: int = 2000,
        min_epochs: int = 10,
        max_epochs: int = 10000,
        num_checkpoint: int = 10,
    ):
        self._dataset_size = dataset_size
        self._target_steps = target_steps
        self._min_epochs = min_epochs
        self._max_epochs = max_epochs
        self._num_checkpoint = num_checkpoint

    def apply_override(self, config: PipelineConfig):
        # TODO(ryand): Use effective batch size here.
        steps_per_epoch = math.ceil(self._dataset_size / config.train_batch_size)
        target_num_epochs = math.ceil(self._target_steps / steps_per_epoch)
        num_epochs = min(max(target_num_epochs, self._min_epochs), self._max_epochs)
        total_steps = num_epochs * steps_per_epoch

        config.max_train_epochs = None
        config.max_train_steps = total_steps

        config.validate_every_n_epochs = None
        config.validate_every_n_steps = total_steps // self._num_checkpoint

        config.save_every_n_epochs = None
        # TODO(ryand): Enable this. During testing, we just want to save images without saving checkpoints to save disk
        # space.
        config.save_every_n_steps = total_steps + 1


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


class AnyFieldOverride(PipelineConfigOverride):
    def __init__(self, field_name: str, value):
        self._field_name = field_name
        self._value = value

    def apply_override(self, config: PipelineConfig):
        field_names = self._field_name.split(".")
        try:
            for field_name in field_names[:-1]:
                config = getattr(config, field_name)
            setattr(config, field_names[-1], self._value)
        except AttributeError:
            raise ValueError(f"Field '{self._field_name}' not found in PipelineConfig.")


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
    placeholder_token: str,
    initializer_token: str,
    validation_prompts: list[str],
    caption_preset: Literal["style", "object"],
    overrides: list[PipelineConfigOverride],
) -> SdxlTextualInversionConfig:
    """Prepare a configuration for training a general SDXL TI model."""

    # TODO: Apply this to all training modes?
    keep_in_memory = dataset_size < 10

    config = SdxlTextualInversionConfig(
        model="stabilityai/stable-diffusion-xl-base-1.0",
        vae_model="madebyollin/sdxl-vae-fp16-fix",
        seed=0,
        base_output_dir="output",
        placeholder_token=placeholder_token,
        initializer_token=initializer_token,
        num_vectors=4,
        optimizer=AdamOptimizerConfig(learning_rate=1e-3),
        lr_scheduler="constant_with_warmup",
        lr_warmup_steps=200,
        mixed_precision="fp16",
        # TODO(ryand): Enable this. During testing, we just want to save images without saving checkpoints to save disk.
        max_checkpoints=1,
        gradient_checkpointing=True,
        validation_prompts=validation_prompts,
        num_validation_images_per_prompt=3,
        train_batch_size=4,
        data_loader=TextualInversionSDDataLoaderConfig(
            dataset=ImageCaptionJsonlDatasetConfig(
                jsonl_path=jsonl_path,
                keep_in_memory=keep_in_memory,
            ),
            caption_preset=caption_preset,
            keep_original_captions=True,
            aspect_ratio_buckets=AspectRatioBucketConfig(
                target_resolution=1024,
                start_dim=512,
                end_dim=1536,
                divisible_by=128,
            ),
            resolution=1024,
            center_crop=True,
            random_flip=False,
            # TODO(ryand): Make this a constant shared by all presets.
            dataloader_num_workers=2,
        ),
    )

    preset_overrides: list[PipelineConfigOverride] = [
        # Configure the training length and checkpoint frequency.
        TrainingLengthOverride(dataset_size),
    ]

    #  Note that we apply the caller-provided overrides before the preset overrides.
    for override in overrides + preset_overrides:
        override.apply_override(config)

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

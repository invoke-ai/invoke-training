from typing import Literal, Optional

from pydantic import BaseModel, Field

from invoke_training.config.pipeline_config import PipelineConfig
from invoke_training.scripts._experimental.presets.config_presets import (
    get_sd_lora_preset_config,
    get_sd_ti_preset_config,
    get_sdxl_lora_preset_config,
    get_sdxl_ti_preset_config,
)


class CommercialConfig(BaseModel):
    jobType: Literal["lora", "ti", "pivotalTuning"]  # noqa: N815
    captionPrefix: str = Field(description="Prefix for captions", default="")  # noqa: N815
    captionPreset: Literal["style", "object"] = Field(description="Caption presets", default=None)  # noqa: N815
    initialPhrase: Optional[str] = Field(description="Initial phrase for text inversion", default=None)  # noqa: N815
    trainingLengthEpochs: int = Field(description="Number of training epochs", default=100)  # noqa: N815
    validationPrompts: list[str] = Field(description="Validation prompts", default=[])  # noqa: N815
    learningRate: float = Field(description="Learning rate", default=1e-4)  # noqa: N815
    modelPath: str = Field(description="Path to the model on disk")  # noqa: N815
    baseModel: Literal["sdxl", "sd-1"] = Field(description="Base model")  # noqa: N815


# TODO(ryand): Confirm that captionPrefix is only set for LORA jobs.


def get_config_preset_from_commercial_config(
    commercial_config: CommercialConfig, jsonl_path: str, dataset_size: int
) -> PipelineConfig:
    if commercial_config.jobType == "lora":
        if commercial_config.baseModel == "sd-1":
            get_sd_lora_preset_config(
                jsonl_path=jsonl_path,
                dataset_size=dataset_size,
                model=commercial_config.modelPath,
                text_encoder_learning_rate=commercial_config.learningRate,
                unet_learning_rate=commercial_config.learningRate,
                caption_prefix=commercial_config.captionPrefix,
                validation_prompts=commercial_config.validationPrompts,
                overrides=[],
            )
        elif commercial_config.baseModel == "sdxl":
            get_sdxl_lora_preset_config(
                jsonl_path=jsonl_path,
                dataset_size=dataset_size,
                model=commercial_config.modelPath,
                # TODO(ryand): Fix SDXL VAE in mixed precision mode.
                vae_model=None,
                text_encoder_learning_rate=commercial_config.learningRate,
                unet_learning_rate=commercial_config.learningRate,
                caption_prefix=commercial_config.captionPrefix,
                validation_prompts=commercial_config.validationPrompts,
                overrides=[],
            )
        else:
            raise ValueError(f"Unsupported base model: {commercial_config.baseModel}")
    elif commercial_config.jobType == "ti":
        if commercial_config.baseModel == "sd-1":
            return get_sd_ti_preset_config(
                jsonl_path=jsonl_path,
                dataset_size=dataset_size,
                model=commercial_config.modelPath,
                # TODO(ryand): How are we handling this?
                placeholder_token="<TOK>",
                # TODO(ryand): Rename initialPhrase to initialToken in the UI.
                initializer_token=commercial_config.initialPhrase,
                # TODO(ryand): We should probably expose num_vectors.
                num_vectors=4,
                learning_rate=commercial_config.learningRate,
                validation_prompts=commercial_config.validationPrompts,
                caption_preset=commercial_config.captionPreset,
                overrides=[],
            )
        elif commercial_config.baseModel == "sdxl":
            return get_sdxl_ti_preset_config(
                jsonl_path=jsonl_path,
                dataset_size=dataset_size,
                model=commercial_config.modelPath,
                # TODO(ryand): Fix SDXL VAE in mixed precision mode.
                vae_model=None,
                # TODO(ryand): How are we handling this?
                placeholder_token="<TOK>",
                # TODO(ryand): Rename initialPhrase to initialToken in the UI.
                initializer_token=commercial_config.initialPhrase,
                # TODO(ryand): We should probably expose num_vectors.
                num_vectors=4,
                learning_rate=commercial_config.learningRate,
                validation_prompts=commercial_config.validationPrompts,
                caption_preset=commercial_config.captionPreset,
                overrides=[],
            )
        else:
            raise ValueError(f"Unsupported base model: {commercial_config.baseModel}")
    else:
        raise ValueError(f"Unsupported job type: {commercial_config.jobType}")

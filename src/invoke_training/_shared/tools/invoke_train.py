from invoke_training.config.pipeline_config import PipelineConfig
from invoke_training.pipelines._experimental.sd_dpo_lora.train import train as train_sd_ddpo_lora
from invoke_training.pipelines.stable_diffusion.lora.train import train as train_sd_lora
from invoke_training.pipelines.stable_diffusion.textual_inversion.train import train as train_sd_ti
from invoke_training.pipelines.stable_diffusion_xl.lora.train import train as train_sdxl_lora
from invoke_training.pipelines.stable_diffusion_xl.lora_and_textual_inversion.train import (
    train as train_sdxl_lora_and_ti,
)
from invoke_training.pipelines.stable_diffusion_xl.textual_inversion.train import train as train_sdxl_ti


def train(config: PipelineConfig):
    if config.type == "SD_LORA":
        train_sd_lora(config)
    elif config.type == "SDXL_LORA":
        train_sdxl_lora(config)
    elif config.type == "SD_TEXTUAL_INVERSION":
        train_sd_ti(config)
    elif config.type == "SDXL_TEXTUAL_INVERSION":
        train_sdxl_ti(config)
    elif config.type == "SDXL_LORA_AND_TEXTUAL_INVERSION":
        train_sdxl_lora_and_ti(config)
    elif config.type == "SD_DIRECT_PREFERENCE_OPTIMIZATION_LORA":
        print(f"Running EXPERIMENTAL pipeline: '{config.type}'.")
        train_sd_ddpo_lora(config)
    else:
        raise ValueError(f"Unexpected pipeline type: '{config.type}'.")

from typing import Annotated, Union

from pydantic import Field

from invoke_training.pipelines._experimental.sd_dpo_lora.config import SdDirectPreferenceOptimizationLoraConfig
from invoke_training.pipelines.stable_diffusion.lora.config import SdLoraConfig
from invoke_training.pipelines.stable_diffusion.textual_inversion.config import SdTextualInversionConfig
from invoke_training.pipelines.stable_diffusion_xl.lora.config import SdxlLoraConfig
from invoke_training.pipelines.stable_diffusion_xl.lora_and_textual_inversion.config import (
    SdxlLoraAndTextualInversionConfig,
)
from invoke_training.pipelines.stable_diffusion_xl.textual_inversion.config import SdxlTextualInversionConfig

PipelineConfig = Annotated[
    Union[
        SdLoraConfig,
        SdxlLoraConfig,
        SdTextualInversionConfig,
        SdxlTextualInversionConfig,
        SdxlLoraAndTextualInversionConfig,
        SdDirectPreferenceOptimizationLoraConfig,
    ],
    Field(discriminator="type"),
]

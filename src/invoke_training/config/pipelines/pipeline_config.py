from typing import Annotated, Union

from pydantic import Field

from invoke_training.config.pipelines.finetune_lora_config import (
    FinetuneLoRASDConfig,
    FinetuneLoRASDXLConfig,
)
from invoke_training.config.pipelines.textual_inversion_config import TextualInversionSDConfig

PipelineConfig = Annotated[
    Union[
        FinetuneLoRASDConfig,
        FinetuneLoRASDXLConfig,
        TextualInversionSDConfig,
    ],
    Field(discriminator="type"),
]

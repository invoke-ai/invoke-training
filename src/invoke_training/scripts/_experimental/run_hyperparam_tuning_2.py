import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from invoke_training._shared.tools.invoke_train import train
from invoke_training.config.optimizer.optimizer_config import AdamOptimizerConfig
from invoke_training.config.pipeline_config import PipelineConfig


@dataclass
class Override:
    base_output_prefix: str
    text_encoder_learning_rate: float
    unet_learning_rate: float

    def get_base_output_dir(self):
        return os.path.join(
            self.base_output_prefix, f"telr_{self.text_encoder_learning_rate}_ulr_{self.unet_learning_rate}"
        )

    def apply(self, config: PipelineConfig):
        config.base_output_dir = self.get_base_output_dir()
        config.optimizer = AdamOptimizerConfig()
        # config.optimizer.optimizer_type = self.optimizer_type
        config.text_encoder_learning_rate = self.text_encoder_learning_rate
        config.unet_learning_rate = self.unet_learning_rate


def run_training(base_config_path, override: Override):
    with open(base_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    pipeline_adapter: TypeAdapter[PipelineConfig] = TypeAdapter(PipelineConfig)
    train_config = pipeline_adapter.validate_python(cfg)

    train_config.max_train_epochs = 50  # Should be more than enough, so we can see progress.
    train_config.save_every_n_epochs = 100  # Intentionally high so that we don't save checkpoints to save disk space.
    train_config.validate_every_n_epochs = 2  # Generate validation images often.

    override.apply(train_config)

    train(train_config)


def main():
    base_config_path = Path(__file__).parent / "configs/sdxl_lora_mohrbacher_1x24gb.yaml"
    name_prefix = "output/hp_tuning/sdxl_lora_mohrbacher_1x24gb/"

    run_overrides = [
        # Effect of varying the learning rate of the text encoder.
        Override(base_output_prefix=name_prefix, text_encoder_learning_rate=1e-3, unet_learning_rate=1e-3),
        Override(base_output_prefix=name_prefix, text_encoder_learning_rate=5e-3, unet_learning_rate=1e-3),
        Override(base_output_prefix=name_prefix, text_encoder_learning_rate=1e-2, unet_learning_rate=1e-3),
        # Effect of varying the learning rate of the UNet.
        Override(base_output_prefix=name_prefix, text_encoder_learning_rate=1e-3, unet_learning_rate=1e-3),
        Override(base_output_prefix=name_prefix, text_encoder_learning_rate=1e-3, unet_learning_rate=5e-3),
        Override(base_output_prefix=name_prefix, text_encoder_learning_rate=1e-3, unet_learning_rate=1e-2),
    ]

    for override in run_overrides:
        run_training(base_config_path, override)


if __name__ == "__main__":
    main()

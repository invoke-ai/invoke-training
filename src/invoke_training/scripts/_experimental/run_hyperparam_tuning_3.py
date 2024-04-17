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
    learning_rate: float
    batch_size: int

    def get_base_output_dir(self):
        return os.path.join(self.base_output_prefix, f"lr_{self.learning_rate}_bs_{self.batch_size}")

    def apply(self, config: PipelineConfig):
        config.base_output_dir = self.get_base_output_dir()
        config.optimizer = AdamOptimizerConfig()
        # config.optimizer.optimizer_type = self.optimizer_type
        config.optimizer.learning_rate = self.learning_rate
        config.train_batch_size = self.batch_size


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
    base_config_path = Path(__file__).parent / "configs/sdxl_ti_apple_vision_pro_1x24gb.yaml"
    name_prefix = "output/hp_tuning/sdxl_ti_avp_1x24gb/"

    run_overrides = [
        # Effect of varying the learning rate.
        Override(base_output_prefix=name_prefix, learning_rate=1e-3, batch_size=1),
        Override(base_output_prefix=name_prefix, learning_rate=2e-3, batch_size=1),
        Override(base_output_prefix=name_prefix, learning_rate=4e-3, batch_size=1),
        # Effect of varying the batch size.
        Override(base_output_prefix=name_prefix, learning_rate=1e-3, batch_size=1),
        Override(base_output_prefix=name_prefix, learning_rate=1e-3, batch_size=2),
        Override(base_output_prefix=name_prefix, learning_rate=1e-3, batch_size=4),
    ]

    for override in run_overrides:
        run_training(base_config_path, override)


if __name__ == "__main__":
    main()

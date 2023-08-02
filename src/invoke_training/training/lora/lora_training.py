import json

from invoke_training.training.lora.lora_training_config import LoRATrainingConfig


def run_lora_training(config: LoRATrainingConfig):
    print(f"Config\n: {json.dumps(config.dict(), indent=2, default=str)}")

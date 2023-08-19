import torch

from invoke_training.training.finetune_lora.finetune_lora_config import (
    TrainingOptimizerConfig,
)


def initialize_optimizer(config: TrainingOptimizerConfig, trainable_params: list) -> torch.optim.Optimizer:
    if config.use_8bit_adam:
        import bitsandbytes

        optimizer_cls = bitsandbytes.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    return optimizer_cls(
        trainable_params,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

import torch
from prodigyopt import Prodigy

from invoke_training.config.shared.optimizer.optimizer_config import OptimizerConfig


def initialize_optimizer(config: OptimizerConfig, trainable_params: list) -> torch.optim.Optimizer:
    """Initialize an optimizer based on the provided config."""

    if config.optimizer.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            weight_decay=config.optimizer.weight_decay,
            eps=config.optimizer.epsilon,
        )
    elif config.optimizer.optimizer_type == "Prodigy":
        optimizer = Prodigy(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.optimizer.weight_decay,
            use_bias_correction=config.optimizer.use_bias_correction,
            safeguard_warmup=config.optimizer.safeguard_warmup,
        )
    else:
        raise ValueError(f"'{config.optimizer}' is not a supported optimizer.")

    return optimizer

import torch
from prodigyopt import Prodigy

from invoke_training.config.shared.optimizer.optimizer_config import AdamOptimizerConfig, ProdigyOptimizerConfig


def initialize_optimizer(
    config: AdamOptimizerConfig | ProdigyOptimizerConfig, trainable_params: list
) -> torch.optim.Optimizer:
    """Initialize an optimizer based on the provided config."""

    if config.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            eps=config.epsilon,
        )
    elif config.optimizer_type == "Prodigy":
        optimizer = Prodigy(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            use_bias_correction=config.use_bias_correction,
            safeguard_warmup=config.safeguard_warmup,
        )
    else:
        raise ValueError(f"'{config.optimizer_type}' is not a supported optimizer.")

    return optimizer

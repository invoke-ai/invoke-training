import typing

from pydantic import BaseModel


class AdamOptimizer(BaseModel):
    optimizer_type: typing.Literal["AdamW"] = "AdamW"

    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-2
    epsilon: float = 1e-8


class ProdigyOptimizer(BaseModel):
    optimizer_type: typing.Literal["Prodigy"] = "Prodigy"

    weight_decay: float = 0.0
    use_bias_correction: bool = False
    safeguard_warmup: bool = False


class OptimizerConfig(BaseModel):
    """Configuration for a training optimizer."""

    optimizer: typing.Union[AdamOptimizer, ProdigyOptimizer] = AdamOptimizer()

    # Initial learning rate to use (after the potential warmup period).
    # Note that this can be overriden for a specific group of params:
    # https://pytorch.org/docs/stable/optim.html#per-parameter-options
    # (E.g. see `text_encoder_learning_rate` and `unet_learning_rate`)
    learning_rate: float = 1e-4

    lr_scheduler: typing.Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ] = "constant"

    # The number of warmup steps in the learning rate scheduler. Only applied to schedulers that support warmup.
    # See lr_scheduler.
    lr_warmup_steps: int = 0

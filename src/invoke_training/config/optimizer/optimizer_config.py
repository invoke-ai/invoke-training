import typing

from invoke_training.config.config_base_model import ConfigBaseModel


class AdamOptimizerConfig(ConfigBaseModel):
    optimizer_type: typing.Literal["AdamW"] = "AdamW"

    learning_rate: float = 1e-4
    """Initial learning rate to use (after the potential warmup period). Note that in some training pipelines this can
    be overriden for a specific group of params: https://pytorch.org/docs/stable/optim.html#per-parameter-options
    (E.g. see `text_encoder_learning_rate` and `unet_learning_rate`)
    """

    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-2
    epsilon: float = 1e-8

    use_8bit: bool = False
    """Use an 8-bit version of the Adam optimizer. This requires the bitsandbytes library to be installed. use_8bit
    reduces the VRAM usage of the optimizer, but increases the risk of issues with numerical stability.
    """


class ProdigyOptimizerConfig(ConfigBaseModel):
    optimizer_type: typing.Literal["Prodigy"] = "Prodigy"

    learning_rate: float = 1.0
    """The learning rate. For the Prodigy optimizer, the learning rate is adjusted dynamically. A value of 1.0 is
    recommended. Note that in some training pipelines this can be overriden for a specific group of params:
    https://pytorch.org/docs/stable/optim.html#per-parameter-options (E.g. see `text_encoder_learning_rate` and
    `unet_learning_rate`)
    """

    weight_decay: float = 0.0
    use_bias_correction: bool = False
    safeguard_warmup: bool = False

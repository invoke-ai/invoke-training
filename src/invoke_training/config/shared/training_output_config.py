import typing

from invoke_training.config.shared.config_base_model import ConfigBaseModel


class TrainingOutputConfig(ConfigBaseModel):
    """Configuration for a training run's output."""

    base_output_dir: str
    """The output directory where the training outputs (model checkpoints, logs, intermediate predictions) will be
    written. A subdirectory will be created with a timestamp for each new training run.
    """

    report_to: typing.Optional[typing.Literal["all", "tensorboard", "wandb", "comet_ml"]] = "tensorboard"
    """The integration to report results and logs to ('all', 'tensorboard', 'wandb', or 'comet_ml'). This value is
    passed to Hugging Face Accelerate. See accelerate.Accelerator.log_with for more details.
    """

    save_model_as: typing.Literal["ckpt", "pt", "safetensors"] = "safetensors"
    """The file type to save the model as. Note that "ckpt" and "pt" are alternative file extensions for the same file
    format.
    """

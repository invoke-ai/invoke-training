import math

from invoke_training.config.pipeline_config import PipelineConfig
from invoke_training.scripts._experimental.presets.pipeline_config_override import PipelineConfigOverride


class TrainingLengthOverride(PipelineConfigOverride):
    """An override to configure the training length and checkpoint frequency.

    This override applies some simple heuristics based on the dataset size to obtain reasonable settings.
    """

    # TODO(ryand): Should there be a max_epochs limit?
    def __init__(
        self,
        dataset_size: int,
        target_steps: int = 2000,
        min_epochs: int = 10,
        max_epochs: int = 10000,
        num_checkpoint: int = 10,
    ):
        self._dataset_size = dataset_size
        self._target_steps = target_steps
        self._min_epochs = min_epochs
        self._max_epochs = max_epochs
        self._num_checkpoint = num_checkpoint

    def apply_override(self, config: PipelineConfig):
        # TODO(ryand): Use effective batch size here.
        steps_per_epoch = math.ceil(self._dataset_size / config.train_batch_size)
        target_num_epochs = math.ceil(self._target_steps / steps_per_epoch)
        num_epochs = min(max(target_num_epochs, self._min_epochs), self._max_epochs)
        total_steps = num_epochs * steps_per_epoch

        config.max_train_epochs = None
        config.max_train_steps = total_steps

        config.validate_every_n_epochs = None
        config.validate_every_n_steps = total_steps // self._num_checkpoint

        config.save_every_n_epochs = None
        # TODO(ryand): Enable this. During testing, we just want to save images without saving checkpoints to save disk
        # space.
        config.save_every_n_steps = total_steps + 1

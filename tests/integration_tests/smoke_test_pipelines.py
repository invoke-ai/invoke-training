import pytest
import yaml
from pydantic import TypeAdapter

from invoke_training.config.pipeline_config import PipelineConfig
from invoke_training.pipelines.callbacks import PipelineCallbacks, TrainingCheckpoint, ValidationImages
from invoke_training.pipelines.invoke_train import train

from ..invoke_training.config.pipelines.config_file_paths import get_pipeline_config_file_paths


class DummyCallbacks(PipelineCallbacks):
    def on_save_checkpoint(self, checkpoint: TrainingCheckpoint):
        assert checkpoint.epoch == 0
        assert checkpoint.step == 1
        assert len(checkpoint.models) >= 1

    def on_save_validation_images(self, images: ValidationImages):
        assert images.epoch == 0
        assert images.step == 1
        assert len(images.images) >= 1


@pytest.mark.fixture("config_file", get_pipeline_config_file_paths())
def test_pipeline(config_file: str):
    """Test that all sample pipeline configs can run for one training step."""

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    pipeline_adapter: TypeAdapter[PipelineConfig] = TypeAdapter(PipelineConfig)

    try:
        pipeline_config = pipeline_adapter.validate_python(cfg)
    except Exception as e:
        raise Exception(f"Error parsing config file: {config_file}") from e

    # Modify pipeline_config to run only one step.
    # TODO ...

    train(pipeline_config, callbacks=[DummyCallbacks()])

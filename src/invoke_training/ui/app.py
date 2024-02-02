import os
import subprocess
import tempfile
import time

import gradio as gr
import yaml

from invoke_training.config.pipeline_config import PipelineConfig
from invoke_training.ui.sd_lora_tab import SdLoraTrainingTab


class App:
    def __init__(self):
        self._config_temp_directory = tempfile.TemporaryDirectory()
        self._training_process = None

        with gr.Blocks() as app:
            SdLoraTrainingTab(run_training_cb=self._run_training, app=app)

        self._app = app

    def launch(self):
        self._app.launch()

    def _run_training(self, config: PipelineConfig):
        # Check if there is already a training process running.
        if self._training_process is not None:
            if self._training_process.poll() is None:
                print(
                    "Tried to start a new training process, but another training process is already running. "
                    "Terminate the existing process first."
                )
                return
            else:
                self._training_process = None

        print(f"Starting {config.type} training...")

        # Write the config to a temporary config file where the training subprocess can read it.
        timestamp = str(time.time()).replace(".", "_")
        config_path = os.path.join(self._config_temp_directory.name, f"{timestamp}.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)

        self._training_process = subprocess.Popen(["invoke-train", "-c", str(config_path)])

        print(f"Started {config.type} training.")

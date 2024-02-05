import os
import subprocess
import tempfile
import time

import gradio as gr
import yaml

from invoke_training.config.pipeline_config import PipelineConfig
from invoke_training.pipelines.stable_diffusion.lora.config import SdLoraConfig
from invoke_training.pipelines.stable_diffusion_xl.lora.config import SdxlLoraConfig
from invoke_training.ui.config_groups.sd_lora_config_group import SdLoraConfigGroup
from invoke_training.ui.config_groups.sdxl_lora_config_group import SdxlLoraConfigGroup
from invoke_training.ui.pipeline_tab import PipelineTab
from invoke_training.ui.utils import get_assets_dir_path, get_config_dir_path


class App:
    def __init__(self):
        self._config_temp_directory = tempfile.TemporaryDirectory()
        self._training_process = None

        logo_path = get_assets_dir_path() / "logo.png"
        with gr.Blocks(analytics_enabled=False) as app:
            with gr.Column():
                gr.Image(value = logo_path, label = "Invoke Training App", width = 200, interactive = False, container = False, )
                with gr.Row():
                    gr.Markdown(
                        "*Invoke Training* - [Documentation](https://invoke-ai.github.io/invoke-training/) -- Learn more about Invoke at [invoke.com](https://www.invoke.com/)"
                    )
            with gr.Tab(label="SD LoRA"):
                PipelineTab(
                    name="SD LoRA",
                    default_config_file_path=str(get_config_dir_path() / "sd_lora_pokemon_1x8gb.yaml"),
                    pipeline_config_cls=SdLoraConfig,
                    config_group_cls=SdLoraConfigGroup,
                    run_training_cb=self._run_training,
                    app=app,
                )
            with gr.Tab(label="SDXL LoRA"):
                PipelineTab(
                    name="SDXL LoRA",
                    default_config_file_path=str(get_config_dir_path() / "sdxl_lora_pokemon_1x24gb.yaml"),
                    pipeline_config_cls=SdxlLoraConfig,
                    config_group_cls=SdxlLoraConfigGroup,
                    run_training_cb=self._run_training,
                    app=app,
                )

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

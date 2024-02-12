import os
import subprocess
import tempfile
import time

import gradio as gr
import yaml

from invoke_training.config.pipeline_config import PipelineConfig
from invoke_training.pipelines.stable_diffusion.lora.config import SdLoraConfig
from invoke_training.pipelines.stable_diffusion.textual_inversion.config import SdTextualInversionConfig
from invoke_training.pipelines.stable_diffusion_xl.lora.config import SdxlLoraConfig
from invoke_training.pipelines.stable_diffusion_xl.lora_and_textual_inversion.config import (
    SdxlLoraAndTextualInversionConfig,
)
from invoke_training.pipelines.stable_diffusion_xl.textual_inversion.config import SdxlTextualInversionConfig
from invoke_training.ui.config_groups.sd_lora_config_group import SdLoraConfigGroup
from invoke_training.ui.config_groups.sd_textual_inversion_config_group import SdTextualInversionConfigGroup
from invoke_training.ui.config_groups.sdxl_lora_and_textual_inversion_config_group import (
    SdxlLoraAndTextualInversionConfigGroup,
)
from invoke_training.ui.config_groups.sdxl_lora_config_group import SdxlLoraConfigGroup
from invoke_training.ui.config_groups.sdxl_textual_inversion_config_group import SdxlTextualInversionConfigGroup
from invoke_training.ui.gradio_blocks.header import Header
from invoke_training.ui.gradio_blocks.pipeline_tab import PipelineTab
from invoke_training.ui.utils import get_config_dir_path


class TrainingPage:
    def __init__(self):
        self._config_temp_directory = tempfile.TemporaryDirectory()
        self._training_process = None

        with gr.Blocks(
            title="invoke-training",
            analytics_enabled=False,
            head='<link rel="icon" type="image/x-icon" href="/assets/favicon.png">',
        ) as app:
            self._header = Header()
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
            with gr.Tab(label="SD Textual Inversion"):
                PipelineTab(
                    name="SD Textual Inversion",
                    default_config_file_path=str(get_config_dir_path() / "sd_textual_inversion_gnome_1x8gb.yaml"),
                    pipeline_config_cls=SdTextualInversionConfig,
                    config_group_cls=SdTextualInversionConfigGroup,
                    run_training_cb=self._run_training,
                    app=app,
                )
            with gr.Tab(label="SDXL Textual Inversion"):
                PipelineTab(
                    name="SDXL Textual Inversion",
                    default_config_file_path=str(get_config_dir_path() / "sdxl_textual_inversion_gnome_1x24gb.yaml"),
                    pipeline_config_cls=SdxlTextualInversionConfig,
                    config_group_cls=SdxlTextualInversionConfigGroup,
                    run_training_cb=self._run_training,
                    app=app,
                )
            with gr.Tab(label="SDXL LoRA and Textual Inversion"):
                PipelineTab(
                    name="SDXL LoRA and Textual Inversion",
                    default_config_file_path=str(get_config_dir_path() / "sdxl_lora_and_ti_gnome_1x24gb.yaml"),
                    pipeline_config_cls=SdxlLoraAndTextualInversionConfig,
                    config_group_cls=SdxlLoraAndTextualInversionConfigGroup,
                    run_training_cb=self._run_training,
                    app=app,
                )

        self._app = app

    def app(self):
        return self._app

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

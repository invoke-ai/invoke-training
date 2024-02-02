import typing

import gradio as gr
import yaml

from invoke_training.config.pipeline_config import PipelineConfig
from invoke_training.pipelines.stable_diffusion.lora.config import SdLoraConfig
from invoke_training.ui.config_groups.sd_lora_config_group import SdLoraConfigGroup
from invoke_training.ui.utils import get_config_dir_path, load_config_from_yaml


class SdLoraTrainingTab:
    def __init__(self, run_training_cb: typing.Callable[[PipelineConfig], None], app: gr.Blocks):
        """The SD_LORA tab for the training app.

        Args:
            run_training_cb (typing.Callable[[PipelineConfig], None]): A callback function to run the training process.
        """
        self._run_training_cb = run_training_cb

        default_config = load_config_from_yaml(self.get_default_config_file_path())
        assert isinstance(default_config, SdLoraConfig)
        # self._default_config is the config that was last loaded from the reference config file.
        self._default_config = default_config
        # self._current_config is the config that was most recently generated from the UI.
        self._current_config = default_config.model_copy(deep=True)

        with gr.Tab(label="SD LoRA"):
            gr.Markdown("# SD LoRA Training Config")
            self.reference_config_file = gr.Textbox(
                label="Reference Config File Path", value=str(self.get_default_config_file_path()), interactive=True
            )
            reset_config_button = gr.Button(value="Reload reference config")
            self.sd_lora_config_group = SdLoraConfigGroup()

            gr.Markdown("## Config Output")
            generate_config_button = gr.Button(value="Generate Config")
            self._config_yaml = gr.Code(label="Config YAML", language="yaml", interactive=False)

            gr.Markdown(
                """# Run Training

                'Start Training' starts the training process in the background. Check the terminal for logs.

                **Warning: Click 'Generate Config' to capture all of the latest changes before starting training.**
                """
            )
            run_training_button = gr.Button(value="Start Training")

            gr.Markdown(
                """# Visualize Results

            Once you've started training, you can see the results by launching tensorboard with the following
            command:

            ```bash
            tensorboard --logdir /path/to/output_dir
            ```

            Alternatively, you can browse the output directory directly to find model checkpoints, logs, and validation
            images.
            """
            )
        reset_config_button.click(
            self.on_reset_config_button_click,
            inputs=self.reference_config_file,
            outputs=self.sd_lora_config_group.get_ui_output_components() + [self._config_yaml],
        )
        generate_config_button.click(
            self.on_generate_config_button_click,
            inputs=set(self.sd_lora_config_group.get_ui_input_components()),
            outputs=self.sd_lora_config_group.get_ui_output_components() + [self._config_yaml],
        )

        run_training_button.click(self.on_run_training_button_click, inputs=[], outputs=[])

        # On app load, reset the configs based on the default reference config file.
        app.load(
            self.on_reset_config_button_click,
            inputs=self.reference_config_file,
            outputs=self.sd_lora_config_group.get_ui_output_components() + [self._config_yaml],
        )

    @classmethod
    def get_default_config_file_path(cls):
        return get_config_dir_path() / "sd_lora_pokemon_1x8gb.yaml"

    def on_reset_config_button_click(self, file_path: str):
        print(f"Resetting UI configs for SD LoRA to {file_path}.")
        default_config = load_config_from_yaml(file_path)
        assert isinstance(default_config, SdLoraConfig)
        self._default_config = default_config
        self._current_config = self._default_config.model_copy(deep=True)
        update_dict = self.sd_lora_config_group.update_ui_components_with_config_data(self._current_config)
        update_dict.update({self._config_yaml: None})
        return update_dict

    def on_generate_config_button_click(self, data: dict):
        print("Generating config for SD LoRA.")
        self._current_config = self.sd_lora_config_group.update_config_with_ui_component_data(
            self._current_config, data
        )

        # Roundtrip to make sure that the config is valid.
        self._current_config = SdLoraConfig.model_validate(self._current_config.model_dump())

        # Update the UI to reflect the new state of the config (in case some values were rounded or otherwise modified
        # in the process).
        update_dict = self.sd_lora_config_group.update_ui_components_with_config_data(self._current_config)
        update_dict.update(
            {
                self._config_yaml: yaml.safe_dump(
                    self._current_config.model_dump(), default_flow_style=False, sort_keys=False
                )
            }
        )
        return update_dict

    def on_run_training_button_click(self):
        self._run_training_cb(self._current_config)

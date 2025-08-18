import typing

import gradio as gr
import yaml

from invoke_training.config.pipeline_config import PipelineConfig
from invoke_training.ui.config_groups.ui_config_element import UIConfigElement
from invoke_training.ui.utils.utils import load_config_from_yaml


class PipelineTab:
    def __init__(
        self,
        name: str,
        default_config_file_path: str,
        pipeline_config_cls: typing.Type[PipelineConfig],
        config_group_cls: typing.Type[UIConfigElement],
        run_training_cb: typing.Callable[[PipelineConfig], None],
        stop_training_cb: typing.Callable[[], str],
        is_training_active_cb: typing.Callable[[], bool],
        app: gr.Blocks,
    ):
        """A tab for a single training pipeline type.

        Args:
            run_training_cb (typing.Callable[[PipelineConfig], None]): A callback function to run the training process.
        """
        self._name = name
        self._default_config_file_path = default_config_file_path
        self._pipeline_config_cls = pipeline_config_cls
        self._run_training_cb = run_training_cb
        self._stop_training_cb = stop_training_cb
        self._is_training_active_cb = is_training_active_cb

        # self._default_config is the config that was last loaded from the reference config file.
        self._default_config = None
        # self._current_config is the config that was most recently generated from the UI.
        self._current_config = None

        gr.Markdown(f"# {self._name} Training Config")
        self.reference_config_file = gr.Textbox(
            label="Reference Config File Path", value=default_config_file_path, interactive=True
        )
        reset_config_button = gr.Button(value="Reload reference config")
        self.pipeline_config_group = config_group_cls()

        gr.Markdown("## Config Output")
        generate_config_button = gr.Button(value="Generate Config")
        self._config_yaml = gr.Code(label="Config YAML", language="yaml", interactive=False)

        gr.Markdown(
            """# Run Training

            'Start Training' starts the training process in the background. Check the terminal for logs.

            **Warning: Click 'Generate Config' to capture all of the latest changes before starting training.**
            """
        )
        
        with gr.Row():
            run_training_button = gr.Button(value="Start Training", variant="primary")
            stop_training_button = gr.Button(value="Stop Training", variant="stop", interactive=False)
        
        with gr.Row():
            training_status = gr.Textbox(
                label="Training Status", 
                value="No training active", 
                interactive=False,
                lines=2
            )

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
            outputs=self.pipeline_config_group.get_ui_output_components() + [self._config_yaml],
        )
        generate_config_button.click(
            self.on_generate_config_button_click,
            inputs=set(self.pipeline_config_group.get_ui_input_components()),
            outputs=self.pipeline_config_group.get_ui_output_components() + [self._config_yaml],
        )

        run_training_button.click(
            self.on_run_training_button_click, 
            inputs=[], 
            outputs=[stop_training_button, training_status]
        )
        stop_training_button.click(
            self.on_stop_training_button_click,
            inputs=[],
            outputs=[run_training_button, stop_training_button, training_status]
        )

        # On app load, reset the configs based on the default reference config file.
        # We'll wrap this in a try-except block to handle any errors during loading
        def safe_load_config(file_path):
            try:
                return self.on_reset_config_button_click(file_path)
            except Exception as e:
                print(f"Error during app.load for {self._name}: {e}")
                # Return empty values for all outputs to avoid UI errors
                output_components = self.pipeline_config_group.get_ui_output_components() + [self._config_yaml]
                return {comp: None for comp in output_components}

        app.load(
            safe_load_config,
            inputs=self.reference_config_file,
            outputs=self.pipeline_config_group.get_ui_output_components() + [self._config_yaml],
        )

        # Add status update function for manual refresh
        def update_training_status():
            is_active = self._is_training_active_cb()
            if is_active:
                return gr.Button(interactive=False), gr.Button(interactive=True), "Training in progress..."
            else:
                return gr.Button(interactive=True), gr.Button(interactive=False), "No training active"

        # Add a refresh button for status updates
        refresh_status_button = gr.Button(value="🔄 Refresh Status", size="sm")
        refresh_status_button.click(
            update_training_status,
            inputs=[],
            outputs=[run_training_button, stop_training_button, training_status]
        )

    def on_reset_config_button_click(self, file_path: str):
        try:
            print(f"Resetting UI configs for {self._name} to {file_path}.")
            default_config = load_config_from_yaml(file_path)

            if not isinstance(default_config, self._pipeline_config_cls):
                raise TypeError(
                    f"Wrong config type. Expected '{self._pipeline_config_cls.__name__}', got "
                    f"'{type(default_config).__name__}'."
                )

            self._default_config = default_config
            self._current_config = self._default_config.model_copy(deep=True)
            update_dict = self.pipeline_config_group.update_ui_components_with_config_data(self._current_config)
            update_dict.update({self._config_yaml: None})
            return update_dict
        except Exception as e:
            print(f"Error resetting config: {e}")
            # Return a minimal update dict to avoid UI errors
            if self._current_config:
                return {
                    self._config_yaml: yaml.safe_dump(
                        self._current_config.model_dump(), default_flow_style=False, sort_keys=False
                    )
                }
            return {self._config_yaml: f"Error loading config: {e}"}

    def on_generate_config_button_click(self, data: dict):
        try:
            print(f"Generating config for {self._name}.")
            self._current_config = self.pipeline_config_group.update_config_with_ui_component_data(
                self._current_config, data
            )

            # Roundtrip to make sure that the config is valid.
            self._current_config = self._pipeline_config_cls.model_validate(self._current_config.model_dump())

            # Update the UI to reflect the new state of the config
            # (in case some values were rounded or otherwise modified
            # in the process).
            update_dict = self.pipeline_config_group.update_ui_components_with_config_data(self._current_config)
            update_dict.update(
                {
                    self._config_yaml: yaml.safe_dump(
                        self._current_config.model_dump(), default_flow_style=False, sort_keys=False
                    )
                }
            )
            return update_dict
        except Exception as e:
            print(f"Error generating config: {e}")
            # Return a minimal update dict to avoid UI errors
            if self._current_config:
                return {
                    self._config_yaml: yaml.safe_dump(
                        self._current_config.model_dump(), default_flow_style=False, sort_keys=False
                    )
                }
            return {self._config_yaml: f"Error generating config: {e}"}

    def on_run_training_button_click(self):
        result = self._run_training_cb(self._current_config)
        if result and "already in progress" in result:
            return gr.Button(interactive=True), result
        else:
            return gr.Button(interactive=False), result or f"Training started for {self._name}..."

    def on_stop_training_button_click(self):
        result = self._stop_training_cb()
        return gr.Button(interactive=True), gr.Button(interactive=False), result

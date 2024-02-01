import gradio as gr

from invoke_training.config.base_pipeline_config import BasePipelineConfig


class BasePipelineConfigGroup:
    def __init__(self):
        self.seed = gr.Number(label="seed", precision=0, interactive=True)
        self.base_output_dir = gr.Textbox(label="base_output_dir", interactive=True)
        with gr.Row():
            self.max_train_steps_or_epochs_dropdown = gr.Dropdown(
                choices=["max_train_steps", "max_train_epochs"], interactive=True
            )
            self.max_train_steps_or_epochs = gr.Number(label="Steps or Epochs", precision=0, interactive=True)

        with gr.Row():
            self.save_every_n_steps_or_epochs_dropdown = gr.Dropdown(
                choices=["save_every_n_steps", "save_every_n_epochs"], interactive=True
            )
            self.save_every_n_steps_or_epochs = gr.Number(label="Steps or Epochs", precision=0, interactive=True)

        with gr.Row():
            self.validate_every_n_steps_or_epochs_dropdown = gr.Dropdown(
                choices=["validate_every_n_steps", "validate_every_n_epochs"], interactive=True
            )
            self.validate_every_n_steps_or_epochs = gr.Number(label="Steps or Epochs", precision=0, interactive=True)

    def get_all_configs(self):
        return [
            self.seed,
            self.base_output_dir,
            self.max_train_steps_or_epochs_dropdown,
            self.max_train_steps_or_epochs,
            self.save_every_n_steps_or_epochs_dropdown,
            self.save_every_n_steps_or_epochs,
            self.validate_every_n_steps_or_epochs_dropdown,
            self.validate_every_n_steps_or_epochs,
        ]

    def update_ui_with_config_data(self, config: BasePipelineConfig):
        if config.max_train_epochs is not None:
            max_train_steps_or_epochs_dropdown = "max_train_epochs"
            max_train_steps_or_epochs = config.max_train_epochs
        elif config.max_train_steps is not None:
            max_train_steps_or_epochs_dropdown = "max_train_steps"
            max_train_steps_or_epochs = config.max_train_steps
        else:
            raise ValueError("One of max_train_epochs or max_train_steps must be set.")

        if config.save_every_n_epochs is not None:
            save_every_n_steps_or_epochs_dropdown = "save_every_n_epochs"
            save_every_n_steps_or_epochs = config.save_every_n_epochs
        elif config.save_every_n_steps is not None:
            save_every_n_steps_or_epochs_dropdown = "save_every_n_steps"
            save_every_n_steps_or_epochs = config.save_every_n_steps
        else:
            raise ValueError("One of save_every_n_epochs or save_every_n_steps must be set.")

        if config.validate_every_n_epochs is not None:
            validate_every_n_steps_or_epochs_dropdown = "validate_every_n_epochs"
            validate_every_n_steps_or_epochs = config.validate_every_n_epochs
        elif config.validate_every_n_steps is not None:
            validate_every_n_steps_or_epochs_dropdown = "validate_every_n_steps"
            validate_every_n_steps_or_epochs = config.validate_every_n_steps
        else:
            raise ValueError("One of validate_every_n_epochs or validate_every_n_steps must be set.")

        return {
            self.seed: config.seed,
            self.base_output_dir: config.base_output_dir,
            self.max_train_steps_or_epochs_dropdown: max_train_steps_or_epochs_dropdown,
            self.max_train_steps_or_epochs: max_train_steps_or_epochs,
            self.save_every_n_steps_or_epochs_dropdown: save_every_n_steps_or_epochs_dropdown,
            self.save_every_n_steps_or_epochs: save_every_n_steps_or_epochs,
            self.validate_every_n_steps_or_epochs_dropdown: validate_every_n_steps_or_epochs_dropdown,
            self.validate_every_n_steps_or_epochs: validate_every_n_steps_or_epochs,
        }

    def update_config_with_ui_data(self, config: BasePipelineConfig, ui_data: dict):
        config.seed = ui_data.pop(self.seed)
        config.base_output_dir = ui_data.pop(self.base_output_dir)

        if ui_data.pop(self.max_train_steps_or_epochs_dropdown) == "max_train_epochs":
            config.max_train_epochs = ui_data.pop(self.max_train_steps_or_epochs)
            config.max_train_steps = None
        else:
            config.max_train_steps = ui_data.pop(self.max_train_steps_or_epochs)
            config.max_train_epochs = None

        if ui_data.pop(self.save_every_n_steps_or_epochs_dropdown) == "save_every_n_epochs":
            config.save_every_n_epochs = ui_data.pop(self.save_every_n_steps_or_epochs)
            config.save_every_n_steps = None
        else:
            config.save_every_n_steps = ui_data.pop(self.save_every_n_steps_or_epochs)
            config.save_every_n_epochs = None

        if ui_data.pop(self.validate_every_n_steps_or_epochs_dropdown) == "validate_every_n_epochs":
            config.validate_every_n_epochs = ui_data.pop(self.validate_every_n_steps_or_epochs)
            config.validate_every_n_steps = None
        else:
            config.validate_every_n_steps = ui_data.pop(self.validate_every_n_steps_or_epochs)
            config.validate_every_n_epochs = None

        return config

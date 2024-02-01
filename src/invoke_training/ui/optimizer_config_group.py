import gradio as gr

from invoke_training.config.optimizer.optimizer_config import AdamOptimizerConfig, ProdigyOptimizerConfig


class OptimizerConfigGroup:
    def __init__(self):
        # TODO(ryand): Add all of the fields.
        self.optimizer = gr.Dropdown(label="optimizer", choices=["AdamW", "Prodigy"], interactive=True)
        self.learning_rate = gr.Number(label="learning_rate", interactive=True)

    def get_all_configs(self):
        return [
            self.optimizer,
            self.learning_rate,
        ]

    def update_ui_with_config_data(self, config: AdamOptimizerConfig | ProdigyOptimizerConfig):
        return {
            self.optimizer: config.optimizer_type,
            self.learning_rate: config.learning_rate,
        }

    def update_config_with_ui_data(self, ui_data: dict) -> AdamOptimizerConfig | ProdigyOptimizerConfig:
        optimizer_type = ui_data.pop(self.optimizer)
        learning_rate = ui_data.pop(self.learning_rate)
        if optimizer_type == "AdamW":
            return AdamOptimizerConfig(optimizer_type=optimizer_type, learning_rate=learning_rate)
        elif optimizer_type == "Prodigy":
            return ProdigyOptimizerConfig(optimizer_type=optimizer_type, learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

import gradio as gr

from invoke_training.config.optimizer.optimizer_config import AdamOptimizerConfig, ProdigyOptimizerConfig


class OptimizerConfigGroup:
    def __init__(self):
        self.optimizer = gr.Dropdown(label="optimizer", choices=["AdamW", "Prodigy"], interactive=True)
        self.learning_rate = gr.Number(label="learning_rate", interactive=True)

    def get_all_configs(self):
        return [
            self.optimizer,
            self.learning_rate,
        ]

    def update_config_state(self, config: AdamOptimizerConfig | ProdigyOptimizerConfig):
        return {
            self.optimizer: config.optimizer_type,
            self.learning_rate: config.learning_rate,
        }

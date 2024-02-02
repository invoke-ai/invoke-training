from typing import Any

import gradio as gr

from invoke_training.config.optimizer.optimizer_config import AdamOptimizerConfig, ProdigyOptimizerConfig
from invoke_training.ui.ui_config_element import UIConfigElement

OptimizerConfig = AdamOptimizerConfig | ProdigyOptimizerConfig


class OptimizerConfigGroup(UIConfigElement):
    def __init__(self):
        self.optimizer = gr.Dropdown(label="optimizer", choices=["AdamW", "Prodigy"], interactive=True)
        self.learning_rate = gr.Number(label="learning_rate", interactive=True)

    def update_ui_components_with_config_data(self, config: OptimizerConfig) -> dict[gr.components.Component, Any]:
        return {
            self.optimizer: config.optimizer_type,
            self.learning_rate: config.learning_rate,
        }

    def update_config_with_ui_component_data(self, orig_config: OptimizerConfig, ui_data: dict) -> OptimizerConfig:
        optimizer_type = ui_data.pop(self.optimizer)
        learning_rate = ui_data.pop(self.learning_rate)
        # TODO: Use the orig_config.
        if optimizer_type == "AdamW":
            return AdamOptimizerConfig(optimizer_type=optimizer_type, learning_rate=learning_rate)
        elif optimizer_type == "Prodigy":
            return ProdigyOptimizerConfig(optimizer_type=optimizer_type, learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

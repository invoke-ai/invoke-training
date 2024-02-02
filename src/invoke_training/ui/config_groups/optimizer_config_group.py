from typing import Any

import gradio as gr

from invoke_training.config.optimizer.optimizer_config import AdamOptimizerConfig, ProdigyOptimizerConfig
from invoke_training.ui.config_groups.ui_config_element import UIConfigElement

OptimizerConfig = AdamOptimizerConfig | ProdigyOptimizerConfig


class AdamOptimizerConfigGroup(UIConfigElement):
    def __init__(self):
        self.learning_rate = gr.Number(label="learning_rate", interactive=True)
        self.beta1 = gr.Number(label="beta1", interactive=True)
        self.beta2 = gr.Number(label="beta2", interactive=True)
        self.weight_decay = gr.Number(label="weight_decay", interactive=True)
        self.epsilon = gr.Number(label="epsilon", interactive=True)

    def update_ui_components_with_config_data(self, config: AdamOptimizerConfig) -> dict[gr.components.Component, Any]:
        return {
            self.learning_rate: config.learning_rate,
            self.beta1: config.beta1,
            self.beta2: config.beta2,
            self.weight_decay: config.weight_decay,
            self.epsilon: config.epsilon,
        }

    def update_config_with_ui_component_data(
        self, orig_config: AdamOptimizerConfig | None, ui_data: dict
    ) -> OptimizerConfig:
        assert orig_config is None

        return AdamOptimizerConfig(
            learning_rate=ui_data.pop(self.learning_rate),
            beta1=ui_data.pop(self.beta1),
            beta2=ui_data.pop(self.beta2),
            weight_decay=ui_data.pop(self.weight_decay),
            epsilon=ui_data.pop(self.epsilon),
        )


class ProdigyOptimizerConfigGroup(UIConfigElement):
    def __init__(self):
        self.learning_rate = gr.Number(label="learning_rate", interactive=True)
        self.weight_decay = gr.Number(label="weight_decay", interactive=True)
        self.use_bias_correction = gr.Checkbox(label="use_bias_correction", interactive=True)
        self.safeguard_warmup = gr.Checkbox(label="safeguard_warmup", interactive=True)

    def update_ui_components_with_config_data(
        self, config: ProdigyOptimizerConfig
    ) -> dict[gr.components.Component, Any]:
        return {
            self.learning_rate: config.learning_rate,
            self.weight_decay: config.weight_decay,
            self.use_bias_correction: config.use_bias_correction,
            self.safeguard_warmup: config.safeguard_warmup,
        }

    def update_config_with_ui_component_data(
        self, orig_config: ProdigyOptimizerConfig | None, ui_data: dict
    ) -> OptimizerConfig:
        assert orig_config is None

        return ProdigyOptimizerConfig(
            learning_rate=ui_data.pop(self.learning_rate),
            weight_decay=ui_data.pop(self.weight_decay),
            use_bias_correction=ui_data.pop(self.use_bias_correction),
            safeguard_warmup=ui_data.pop(self.safeguard_warmup),
        )


class OptimizerConfigGroup(UIConfigElement):
    def __init__(self):
        self.optimizer_type = gr.Dropdown(label="optimizer", choices=["AdamW", "Prodigy"], interactive=True)

        with gr.Group() as adam_optimizer_config_group:
            self.adam_optimizer_config = AdamOptimizerConfigGroup()
        self.adam_optimizer_config_group = adam_optimizer_config_group

        with gr.Group() as prodigy_optimizer_config_group:
            self.prodigy_optimizer_config = ProdigyOptimizerConfigGroup()
        self.prodigy_optimizer_config_group = prodigy_optimizer_config_group

        self.optimizer_type.change(
            self._on_optimizer_type_change,
            inputs=[self.optimizer_type],
            outputs=[self.adam_optimizer_config_group, self.prodigy_optimizer_config_group],
        )

    def _on_optimizer_type_change(self, optimizer_type: str):
        return {
            self.adam_optimizer_config_group: gr.Group(visible=optimizer_type == "AdamW"),
            self.prodigy_optimizer_config_group: gr.Group(visible=optimizer_type == "Prodigy"),
        }

    def update_ui_components_with_config_data(self, config: OptimizerConfig) -> dict[gr.components.Component, Any]:
        update_dict = {
            self.optimizer_type: config.optimizer_type,
            self.adam_optimizer_config_group: gr.Group(visible=config.optimizer_type == "AdamW"),
            self.prodigy_optimizer_config_group: gr.Group(visible=config.optimizer_type == "Prodigy"),
        }

        update_dict.update(
            self.adam_optimizer_config.update_ui_components_with_config_data(
                config if config.optimizer_type == "AdamW" else AdamOptimizerConfig()
            )
        )
        update_dict.update(
            self.prodigy_optimizer_config.update_ui_components_with_config_data(
                config if config.optimizer_type == "Prodigy" else ProdigyOptimizerConfig()
            )
        )

        return update_dict

    def update_config_with_ui_component_data(self, orig_config: OptimizerConfig, ui_data: dict) -> OptimizerConfig:
        # TODO: Use orig_config?

        new_config_adam = self.adam_optimizer_config.update_config_with_ui_component_data(None, ui_data)
        new_config_prodigy = self.prodigy_optimizer_config.update_config_with_ui_component_data(None, ui_data)

        optimizer_type = ui_data.pop(self.optimizer_type)
        if optimizer_type == "AdamW":
            return new_config_adam
        elif optimizer_type == "Prodigy":
            return new_config_prodigy
        else:
            raise ValueError(f"Invalid optimizer type: {optimizer_type}")

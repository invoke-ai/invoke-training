import typing

import gradio as gr

from invoke_training.pipelines.stable_diffusion.lora.config import SdLoraConfig
from invoke_training.ui.config_groups.base_pipeline_config_group import BasePipelineConfigGroup
from invoke_training.ui.config_groups.image_caption_sd_data_loader_config_group import (
    ImageCaptionSDDataLoaderConfigGroup,
)
from invoke_training.ui.config_groups.optimizer_config_group import OptimizerConfigGroup
from invoke_training.ui.config_groups.ui_config_element import UIConfigElement
from invoke_training.ui.utils import get_typing_literal_options


class SdLoraConfigGroup(UIConfigElement):
    def __init__(self):
        """The SD_LORA configs."""

        gr.Markdown("## Basic Configs")
        self.model = gr.Textbox(label="model", info="TODO: Add more info", type="text", interactive=True)
        self.hf_variant = gr.Textbox(label="hf_variant", type="text", interactive=True)
        self.base_pipeline_config_group = BasePipelineConfigGroup()
        self.max_checkpoints = gr.Number(label="max_checkpoints", interactive=True, precision=0)

        gr.Markdown("## Data Configs")
        # TODO: This element should support all data loader types.
        self.image_caption_sd_data_loader_config_group = ImageCaptionSDDataLoaderConfigGroup()

        gr.Markdown("## Optimization Configs")
        self.optimizer_config_group = OptimizerConfigGroup()
        with gr.Row():
            self.train_unet = gr.Checkbox(label="train_unet", interactive=True)
            self.unet_learning_rate = gr.Number(label="unet_learning_rate", interactive=True)
        with gr.Row():
            self.train_text_encoder = gr.Checkbox(label="train_text_encoder", interactive=True)
            self.text_encoder_learning_rate = gr.Number(label="text_encoder_learning_rate", interactive=True)
        self.lr_scheduler = gr.Dropdown(
            label="lr_scheduler",
            choices=get_typing_literal_options(SdLoraConfig, "lr_scheduler"),
            interactive=True,
        )
        self.lr_warmup_steps = gr.Number(label="lr_warmup_steps", interactive=True)
        self.max_grad_norm = gr.Number(label="max_grad_norm", interactive=True)
        self.train_batch_size = gr.Number(label="train_batch_size", precision=0, interactive=True)

        gr.Markdown("## Speed / Memory Configs")
        self.cache_text_encoder_outputs = gr.Checkbox(label="cache_text_encoder_outputs", interactive=True)
        self.cache_vae_outputs = gr.Checkbox(label="cache_vae_outputs", interactive=True)
        self.enable_cpu_offload_during_validation = gr.Checkbox(
            label="enable_cpu_offload_during_validation", interactive=True
        )
        self.gradient_accumulation_steps = gr.Number(label="gradient_accumulation_steps", precision=0, interactive=True)
        self.mixed_precision = gr.Dropdown(
            label="mixed_precision",
            choices=get_typing_literal_options(SdLoraConfig, "mixed_precision"),
            interactive=True,
        )
        self.gradient_checkpointing = gr.Checkbox(label="gradient_checkpointing", interactive=True)

        gr.Markdown("## Model Configs")
        self.lora_rank_dim = gr.Number(label="lora_rank_dim", interactive=True, precision=0)
        self.min_snr_gamma = gr.Number(label="min_snr_gamma", interactive=True)

        gr.Markdown("## Validation")
        self.validation_prompts = gr.Textbox(
            label="validation_prompts", info="Enter one validation prompt per line.", lines=5, interactive=True
        )
        self.num_validation_images_per_prompt = gr.Number(
            label="num_validation_images_per_prompt", precision=0, interactive=True
        )

    def update_ui_components_with_config_data(self, config: SdLoraConfig) -> dict[gr.components.Component, typing.Any]:
        update_dict = {
            self.model: config.model,
            self.hf_variant: config.hf_variant,
            self.max_checkpoints: config.max_checkpoints,
            self.train_unet: config.train_unet,
            self.unet_learning_rate: config.unet_learning_rate,
            self.train_text_encoder: config.train_text_encoder,
            self.text_encoder_learning_rate: config.text_encoder_learning_rate,
            self.lr_scheduler: config.lr_scheduler,
            self.lr_warmup_steps: config.lr_warmup_steps,
            self.max_grad_norm: config.max_grad_norm,
            self.train_batch_size: config.train_batch_size,
            self.cache_text_encoder_outputs: config.cache_text_encoder_outputs,
            self.cache_vae_outputs: config.cache_vae_outputs,
            self.enable_cpu_offload_during_validation: config.enable_cpu_offload_during_validation,
            self.gradient_accumulation_steps: config.gradient_accumulation_steps,
            self.mixed_precision: config.mixed_precision,
            self.gradient_checkpointing: config.gradient_checkpointing,
            self.lora_rank_dim: config.lora_rank_dim,
            self.min_snr_gamma: config.min_snr_gamma,
            self.validation_prompts: "\n".join(config.validation_prompts),
            self.num_validation_images_per_prompt: config.num_validation_images_per_prompt,
        }
        update_dict.update(
            self.image_caption_sd_data_loader_config_group.update_ui_components_with_config_data(config.data_loader)
        )
        update_dict.update(self.base_pipeline_config_group.update_ui_components_with_config_data(config))
        update_dict.update(self.optimizer_config_group.update_ui_components_with_config_data(config.optimizer))

        # Sanity check to catch if we accidentally forget to update a UI component.
        assert set(update_dict.keys()) == set(self.get_ui_output_components())

        return update_dict

    def update_config_with_ui_component_data(
        self, orig_config: SdLoraConfig, ui_data: dict[gr.components.Component, typing.Any]
    ) -> SdLoraConfig:
        new_config = orig_config.model_copy(deep=True)

        new_config.model = ui_data.pop(self.model)
        new_config.hf_variant = ui_data.pop(self.hf_variant)
        new_config.max_checkpoints = ui_data.pop(self.max_checkpoints)
        new_config.train_unet = ui_data.pop(self.train_unet)
        new_config.unet_learning_rate = ui_data.pop(self.unet_learning_rate)
        new_config.train_text_encoder = ui_data.pop(self.train_text_encoder)
        new_config.text_encoder_learning_rate = ui_data.pop(self.text_encoder_learning_rate)
        new_config.lr_scheduler = ui_data.pop(self.lr_scheduler)
        new_config.lr_warmup_steps = ui_data.pop(self.lr_warmup_steps)
        new_config.max_grad_norm = ui_data.pop(self.max_grad_norm)
        new_config.train_batch_size = ui_data.pop(self.train_batch_size)
        new_config.cache_text_encoder_outputs = ui_data.pop(self.cache_text_encoder_outputs)
        new_config.cache_vae_outputs = ui_data.pop(self.cache_vae_outputs)
        new_config.enable_cpu_offload_during_validation = ui_data.pop(self.enable_cpu_offload_during_validation)
        new_config.gradient_accumulation_steps = ui_data.pop(self.gradient_accumulation_steps)
        new_config.mixed_precision = ui_data.pop(self.mixed_precision)
        new_config.gradient_checkpointing = ui_data.pop(self.gradient_checkpointing)
        new_config.lora_rank_dim = ui_data.pop(self.lora_rank_dim)
        new_config.min_snr_gamma = ui_data.pop(self.min_snr_gamma)
        new_config.num_validation_images_per_prompt = ui_data.pop(self.num_validation_images_per_prompt)

        validation_prompts: list[str] = ui_data.pop(self.validation_prompts).split("\n")
        validation_prompts = [x.strip() for x in validation_prompts if x.strip() != ""]
        new_config.validation_prompts = validation_prompts

        new_config.data_loader = self.image_caption_sd_data_loader_config_group.update_config_with_ui_component_data(
            new_config.data_loader, ui_data
        )
        new_config = self.base_pipeline_config_group.update_config_with_ui_component_data(new_config, ui_data)
        new_config.optimizer = self.optimizer_config_group.update_config_with_ui_component_data(
            new_config.optimizer, ui_data
        )

        # We pop items from ui_data as we use them so that we can sanity check that all the input data was transferred
        # to the config.
        assert len(ui_data) == 0

        return new_config

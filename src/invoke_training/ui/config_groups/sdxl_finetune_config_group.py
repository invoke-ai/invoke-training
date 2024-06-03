import typing

import gradio as gr

from invoke_training.pipelines.stable_diffusion_xl.finetune.config import SdxlFinetuneConfig
from invoke_training.ui.config_groups.base_pipeline_config_group import BasePipelineConfigGroup
from invoke_training.ui.config_groups.image_caption_sd_data_loader_config_group import (
    ImageCaptionSDDataLoaderConfigGroup,
)
from invoke_training.ui.config_groups.optimizer_config_group import OptimizerConfigGroup
from invoke_training.ui.config_groups.ui_config_element import UIConfigElement
from invoke_training.ui.utils.prompts import (
    convert_pos_neg_prompts_to_ui_prompts,
    convert_ui_prompts_to_pos_neg_prompts,
)
from invoke_training.ui.utils.utils import get_typing_literal_options


class SdxlFinetuneConfigGroup(UIConfigElement):
    def __init__(self):
        """The SDXL_FINETUNE configs."""

        gr.Markdown("## Basic Configs")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("Base Model"):
                    self.model = gr.Textbox(
                        label="Model",
                        info="The base model. Can be a Hugging Face Hub model name, or a path to a local model (in "
                        "diffusers or checkpoint format).",
                        type="text",
                        interactive=True,
                    )
                    self.hf_variant = gr.Textbox(
                        label="Variant",
                        info="(optional) The Hugging Face hub model variant (e.g., fp16, fp32) to use if the model is a"
                        " HF Hub model name.",
                        type="text",
                        interactive=True,
                    )
                    self.vae_model = gr.Textbox(
                        label="VAE Model",
                        info="(optional) If set, this overrides the base model's default VAE model.",
                        type="text",
                        interactive=True,
                    )
            with gr.Column(scale=3):
                with gr.Tab("Training Outputs"):
                    self.base_pipeline_config_group = BasePipelineConfigGroup()
                    self.save_checkpoint_format = gr.Dropdown(
                        label="Checkpoint Format",
                        info="The save format for the checkpoints. `full_diffusers` saves the full model in diffusers "
                        "format. `trained_only_diffusers` saves only the parts of the model that were finetuned "
                        "(i.e. the UNet).",
                        choices=get_typing_literal_options(SdxlFinetuneConfig, "save_checkpoint_format"),
                        interactive=True,
                    )
                    self.save_dtype = gr.Dropdown(
                        label="Save Dtype",
                        info="The dtype to use when saving the model.",
                        choices=get_typing_literal_options(SdxlFinetuneConfig, "save_dtype"),
                        interactive=True,
                    )
                    self.max_checkpoints = gr.Number(
                        label="Maximum Number of Checkpoints",
                        info="The maximum number of checkpoints to keep on disk from this training run. Earlier "
                        "checkpoints will be deleted to respect this limit.",
                        interactive=True,
                        precision=0,
                    )

        gr.Markdown("## Data Configs")
        self.image_caption_sd_data_loader_config_group = ImageCaptionSDDataLoaderConfigGroup()

        gr.Markdown("## Optimizer Configs")
        self.optimizer_config_group = OptimizerConfigGroup()

        gr.Markdown("## Speed / Memory Configs")
        with gr.Group():
            with gr.Row():
                self.gradient_accumulation_steps = gr.Number(
                    label="Gradient Accumulation Steps",
                    info="The number of gradient steps to accumulate before each weight update. This is an alternative"
                    "to increasing the batch size when training with limited VRAM."
                    "effective_batch_size = train_batch_size * gradient_accumulation_steps.",
                    precision=0,
                    interactive=True,
                )
            with gr.Row():
                self.weight_dtype = gr.Dropdown(
                    label="Weight Type",
                    info="The precision of the model weights. Lower precision can speed up training and reduce memory, "
                    "with increased risk of numerical stability issues. 'bfloat16' is recommended for most use cases "
                    "if your GPU supports it.",
                    choices=get_typing_literal_options(SdxlFinetuneConfig, "weight_dtype"),
                    interactive=True,
                )
            with gr.Row():
                self.cache_text_encoder_outputs = gr.Checkbox(
                    label="Cache Text Encoder Outputs",
                    info="Cache the text encoder outputs to increase speed. This should not be used when training the "
                    "text encoder or performing data augmentations that would change the text encoder outputs.",
                    interactive=True,
                )
                self.cache_vae_outputs = gr.Checkbox(
                    label="Cache VAE Outputs",
                    info="Cache the VAE outputs to increase speed. This should not be used when training the UNet or "
                    "performing data augmentations that would change the VAE outputs.",
                    interactive=True,
                )
            with gr.Row():
                self.enable_cpu_offload_during_validation = gr.Checkbox(
                    label="Enable CPU Offload during Validation",
                    info="Offload models to the CPU sequentially during validation. This reduces peak VRAM "
                    "requirements at the cost of slower validation during training.",
                    interactive=True,
                )
                self.gradient_checkpointing = gr.Checkbox(
                    label="Gradient Checkpointing",
                    info="If True, VRAM requirements are reduced at the cost of ~20% slower training",
                    interactive=True,
                )

        gr.Markdown("## General Training Configs")
        with gr.Tab("Core"):
            with gr.Row():
                self.lr_scheduler = gr.Dropdown(
                    label="Learning Rate Scheduler",
                    choices=get_typing_literal_options(SdxlFinetuneConfig, "lr_scheduler"),
                    interactive=True,
                )
                self.lr_warmup_steps = gr.Number(
                    label="Warmup Steps",
                    info="The number of warmup steps in the "
                    "learning rate schedule, if applicable to the selected scheduler.",
                    interactive=True,
                )
            with gr.Row():
                self.use_masks = gr.Checkbox(
                    label="Use Masks", info="This can only be enabled if the dataset contains masks.", interactive=True
                )

        with gr.Tab("Advanced"):
            with gr.Row():
                self.min_snr_gamma = gr.Number(
                    label="Minimum SNR Gamma",
                    info="min_snr_gamma acts like an an upper bound on the weight of samples with low noise "
                    "levels. If None, then Min-SNR weighting will not be applied. If enabled, the recommended "
                    "value is min_snr gamma = 5.0.",
                    interactive=True,
                )
                self.max_grad_norm = gr.Number(
                    label="Max Gradient Norm",
                    info="Max gradient norm for clipping. Set to None for no clipping.",
                    interactive=True,
                )
                self.train_batch_size = gr.Number(
                    label="Batch Size",
                    info="The Training Batch Size - Higher values require increasing amounts of VRAM.",
                    precision=0,
                    interactive=True,
                )

        gr.Markdown("## Validation")
        with gr.Group():
            self.validation_prompts = gr.Textbox(
                label="Validation Prompts",
                info="Enter one validation prompt per line. Optionally, add negative prompts after a '[NEG]' "
                "delimiter. For example: `positive prompt[NEG]negative prompt`. ",
                lines=5,
                interactive=True,
            )
            self.num_validation_images_per_prompt = gr.Number(
                label="# of Validation Images to Generate per Prompt", precision=0, interactive=True
            )

    def update_ui_components_with_config_data(
        self, config: SdxlFinetuneConfig
    ) -> dict[gr.components.Component, typing.Any]:
        update_dict = {
            self.model: config.model,
            self.hf_variant: config.hf_variant,
            self.vae_model: config.vae_model,
            self.save_checkpoint_format: config.save_checkpoint_format,
            self.save_dtype: config.save_dtype,
            self.max_checkpoints: config.max_checkpoints,
            self.lr_scheduler: config.lr_scheduler,
            self.lr_warmup_steps: config.lr_warmup_steps,
            self.use_masks: config.use_masks,
            self.min_snr_gamma: config.min_snr_gamma,
            self.max_grad_norm: config.max_grad_norm,
            self.train_batch_size: config.train_batch_size,
            self.cache_text_encoder_outputs: config.cache_text_encoder_outputs,
            self.cache_vae_outputs: config.cache_vae_outputs,
            self.enable_cpu_offload_during_validation: config.enable_cpu_offload_during_validation,
            self.gradient_accumulation_steps: config.gradient_accumulation_steps,
            self.weight_dtype: config.weight_dtype,
            self.gradient_checkpointing: config.gradient_checkpointing,
            self.validation_prompts: convert_pos_neg_prompts_to_ui_prompts(
                config.validation_prompts, config.negative_validation_prompts
            ),
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
        self, orig_config: SdxlFinetuneConfig, ui_data: dict[gr.components.Component, typing.Any]
    ) -> SdxlFinetuneConfig:
        new_config = orig_config.model_copy(deep=True)

        new_config.model = ui_data.pop(self.model)
        new_config.hf_variant = ui_data.pop(self.hf_variant) or None
        new_config.vae_model = ui_data.pop(self.vae_model) or None
        new_config.save_checkpoint_format = ui_data.pop(self.save_checkpoint_format)
        new_config.save_dtype = ui_data.pop(self.save_dtype)
        new_config.max_checkpoints = ui_data.pop(self.max_checkpoints)
        new_config.lr_scheduler = ui_data.pop(self.lr_scheduler)
        new_config.lr_warmup_steps = ui_data.pop(self.lr_warmup_steps)
        new_config.use_masks = ui_data.pop(self.use_masks)
        new_config.min_snr_gamma = ui_data.pop(self.min_snr_gamma)
        new_config.max_grad_norm = ui_data.pop(self.max_grad_norm)
        new_config.train_batch_size = ui_data.pop(self.train_batch_size)
        new_config.cache_text_encoder_outputs = ui_data.pop(self.cache_text_encoder_outputs)
        new_config.cache_vae_outputs = ui_data.pop(self.cache_vae_outputs)
        new_config.enable_cpu_offload_during_validation = ui_data.pop(self.enable_cpu_offload_during_validation)
        new_config.gradient_accumulation_steps = ui_data.pop(self.gradient_accumulation_steps)
        new_config.weight_dtype = ui_data.pop(self.weight_dtype)
        new_config.gradient_checkpointing = ui_data.pop(self.gradient_checkpointing)
        new_config.num_validation_images_per_prompt = ui_data.pop(self.num_validation_images_per_prompt)

        positive_prompts, negative_prompts = convert_ui_prompts_to_pos_neg_prompts(ui_data.pop(self.validation_prompts))
        new_config.validation_prompts = positive_prompts
        new_config.negative_validation_prompts = negative_prompts

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
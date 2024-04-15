import typing

import gradio as gr

from invoke_training.pipelines.stable_diffusion_xl.lora_and_textual_inversion.config import (
    SdxlLoraAndTextualInversionConfig,
)
from invoke_training.ui.config_groups.base_pipeline_config_group import BasePipelineConfigGroup
from invoke_training.ui.config_groups.optimizer_config_group import OptimizerConfigGroup
from invoke_training.ui.config_groups.textual_inversion_sd_data_loader_config_group import (
    TextualInversionSDDataLoaderConfigGroup,
)
from invoke_training.ui.config_groups.ui_config_element import UIConfigElement
from invoke_training.ui.utils.prompts import (
    convert_pos_neg_prompts_to_ui_prompts,
    convert_ui_prompts_to_pos_neg_prompts,
)
from invoke_training.ui.utils.utils import get_typing_literal_options


class SdxlLoraAndTextualInversionConfigGroup(UIConfigElement):
    def __init__(self):
        """The SDXL_LORA_AND_TEXTUAL_INVERSION configs."""

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
                    self.max_checkpoints = gr.Number(
                        label="Maximum Number of Checkpoints",
                        info="The maximum number of checkpoints to keep on disk from this training run. Earlier "
                        "checkpoints will be deleted to respect this limit.",
                        interactive=True,
                        precision=0,
                    )

        gr.Markdown("## Data Configs")
        self.image_caption_sd_data_loader_config_group = TextualInversionSDDataLoaderConfigGroup()

        gr.Markdown("## Textual Inversion Configs")
        self.num_vectors = gr.Number(
            label="Num Vectors",
            info="The number of TI vectors that will be trained. Can be overriden by 'Initial Phrase'.",
            interactive=True,
            precision=0,
        )
        self.placeholder_token = gr.Textbox(
            label="Placeholder Token",
            info="The special word to associate the learned embeddings with. Choose a unique token that is unlikely to "
            "already exist in the tokenizer's vocabulary.",
            interactive=True,
        )
        self.initializer_token = gr.Textbox(
            label="Initializer Token",
            info="Only one of 'Initializer Token' or 'Initial Phrase' should be set. A vocabulary token to use as an "
            "initializer for the placeholder token. It should be a single word that roughly describes the object or "
            "style that you're trying to train on. The initializer token ust map to a single tokenizer token.",
            interactive=True,
        )
        self.initial_phrase = gr.Textbox(
            label="Initial Phrase",
            info="Only one of 'Initializer Token' or 'Initial Phrase' should be set. A phrase that will be used to "
            "initialize the placeholder token embedding. The phrase will be tokenized, and the corresponding "
            "embeddings will be used to initialize the placeholder tokens. The number of embedding vectors will be "
            "inferred from the length of the tokenized phrase, so keep the phrase short.",
            interactive=True,
        )

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
                self.mixed_precision = gr.Dropdown(
                    label="Mixed Precision",
                    info="The mixed precision training mode to used. Using a lower precision can speed up training and "
                    "reduce memory usage, with a minor quality hit.",
                    choices=get_typing_literal_options(SdxlLoraAndTextualInversionConfig, "mixed_precision"),
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
                self.train_unet = gr.Checkbox(label="Train UNet", interactive=True)
                self.train_text_encoder = gr.Checkbox(label="Train Text Encoder", interactive=True)
                self.train_ti = gr.Checkbox(label="Train Textual Inversion Token", scale=2, interactive=True)
            with gr.Row():
                self.unet_learning_rate = gr.Number(label="UNet Learning Rate", interactive=True)
                self.text_encoder_learning_rate = gr.Number(label="Text Encoder Learning Rate", interactive=True)
                self.textual_inversion_learning_rate = gr.Number(
                    label="Textual Inversion Learning Rate", interactive=True
                )
                self.ti_train_steps_ratio = gr.Number(label="Textual Inversion Train Steps Ratio", interactive=True)
            with gr.Row():
                self.lr_scheduler = gr.Dropdown(
                    label="Learning Rate Scheduler",
                    choices=get_typing_literal_options(SdxlLoraAndTextualInversionConfig, "lr_scheduler"),
                    interactive=True,
                )
                self.lr_warmup_steps = gr.Number(
                    label="Warmup Steps",
                    info="The number of warmup steps in the "
                    "learning rate schedule, if applicable to the selected scheduler.",
                    interactive=True,
                )
        with gr.Tab("Advanced"):
            with gr.Column():
                self.lora_rank_dim = gr.Number(
                    label="LoRA Rank Dim",
                    info="The rank dimension to use for the LoRA layers. Increasing the rank dimension increases"
                    " the model's expressivity, but also increases the size of the generated LoRA model.",
                    interactive=True,
                    precision=0,
                )
                self.min_snr_gamma = gr.Number(
                    label="Minumum SNR Gamma",
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
        self, config: SdxlLoraAndTextualInversionConfig
    ) -> dict[gr.components.Component, typing.Any]:
        update_dict = {
            self.model: config.model,
            self.hf_variant: config.hf_variant,
            self.vae_model: config.vae_model,
            self.num_vectors: config.num_vectors,
            self.placeholder_token: config.placeholder_token,
            self.initializer_token: config.initializer_token,
            self.initial_phrase: config.initial_phrase,
            self.max_checkpoints: config.max_checkpoints,
            self.train_unet: config.train_unet,
            self.train_text_encoder: config.train_text_encoder,
            self.train_ti: config.train_ti,
            self.unet_learning_rate: config.unet_learning_rate,
            self.text_encoder_learning_rate: config.text_encoder_learning_rate,
            self.textual_inversion_learning_rate: config.textual_inversion_learning_rate,
            self.ti_train_steps_ratio: config.ti_train_steps_ratio,
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
        self, orig_config: SdxlLoraAndTextualInversionConfig, ui_data: dict[gr.components.Component, typing.Any]
    ) -> SdxlLoraAndTextualInversionConfig:
        new_config = orig_config.model_copy(deep=True)

        new_config.model = ui_data.pop(self.model)
        new_config.hf_variant = ui_data.pop(self.hf_variant) or None
        new_config.vae_model = ui_data.pop(self.vae_model) or None
        new_config.num_vectors = ui_data.pop(self.num_vectors)
        new_config.placeholder_token = ui_data.pop(self.placeholder_token)
        new_config.initializer_token = ui_data.pop(self.initializer_token) or None
        new_config.initial_phrase = ui_data.pop(self.initial_phrase) or None
        new_config.max_checkpoints = ui_data.pop(self.max_checkpoints)
        new_config.train_unet = ui_data.pop(self.train_unet)
        new_config.train_text_encoder = ui_data.pop(self.train_text_encoder)
        new_config.train_ti = ui_data.pop(self.train_ti)
        new_config.unet_learning_rate = ui_data.pop(self.unet_learning_rate)
        new_config.text_encoder_learning_rate = ui_data.pop(self.text_encoder_learning_rate)
        new_config.textual_inversion_learning_rate = ui_data.pop(self.textual_inversion_learning_rate)
        new_config.ti_train_steps_ratio = ui_data.pop(self.ti_train_steps_ratio)
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

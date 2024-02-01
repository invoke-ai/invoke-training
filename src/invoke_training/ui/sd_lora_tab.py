import typing

import gradio as gr
import yaml

from invoke_training.config.pipeline_config import PipelineConfig
from invoke_training.pipelines.stable_diffusion.lora.config import SdLoraConfig
from invoke_training.ui.base_pipeline_config_group import BasePipelineConfigGroup
from invoke_training.ui.optimizer_config_group import OptimizerConfigGroup
from invoke_training.ui.utils import get_config_dir_path, get_typing_literal_options, load_config_from_yaml


class SdLoraTrainingTab:
    def __init__(self, run_training_cb: typing.Callable[[PipelineConfig], None]):
        """The SD_LORA tab for the training app.

        Args:
            run_training_cb (typing.Callable[[PipelineConfig], None]): A callback function to run the training process.
        """
        self._run_training_cb = run_training_cb

        default_config = load_config_from_yaml(self.get_default_config_file_path())
        assert isinstance(default_config, SdLoraConfig)
        self._default_config = default_config
        self._current_config = default_config.model_copy(deep=True)

        with gr.Tab(label="SD LoRA"):
            gr.Markdown("# SD LoRA Training Config")
            reset_config_defaults_button = gr.Button(value="Reset Config Defaults")

            gr.Markdown("## Basic Configs")
            self.model = gr.Textbox(label="model", info="TODO: Add more info", type="text", interactive=True)
            self.hf_variant = gr.Textbox(label="hf_variant", type="text", interactive=True)
            self.base_pipeline_config_group = BasePipelineConfigGroup()

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
            self.gradient_accumulation_steps = gr.Number(
                label="gradient_accumulation_steps", precision=0, interactive=True
            )
            self.mixed_precision = gr.Dropdown(
                label="mixed_precision",
                choices=get_typing_literal_options(SdLoraConfig, "mixed_precision"),
                interactive=True,
            )
            self.gradient_checkpointing = gr.Checkbox(label="gradient_checkpointing", interactive=True)

            gr.Markdown("## LoRA Configs")
            self.lora_rank_dim = gr.Number(label="lora_rank_dim", interactive=True, precision=0)

            gr.Markdown("## Validation")
            self.validation_prompts = gr.Textbox(
                label="validation_prompts", info="Enter one validation prompt per line.", lines=5, interactive=True
            )
            self.num_validation_images_per_prompt = gr.Number(
                label="num_validation_images_per_prompt", precision=0, interactive=True
            )

            gr.Markdown("## Config Output")
            generate_config_button = gr.Button(value="Generate Config")
            self._config_yaml = gr.Code(label="Config YAML", language="yaml", interactive=False)

            gr.Markdown("## Run Training")
            gr.Markdown("'Start Training' starts the training process in the background. Check the terminal for logs.")
            run_training_button = gr.Button(value="Start Training")

        reset_config_defaults_button.click(
            self.on_reset_config_defaults_button_click, inputs=[], outputs=self.get_all_configs()
        )
        generate_config_button.click(
            self.on_generate_config_button_click,
            inputs=set(self.get_all_configs()),
            outputs=self.get_all_configs() + [self._config_yaml],
        )
        run_training_button.click(self.on_run_training_button_click, inputs=[], outputs=[])

    @classmethod
    def get_default_config_file_path(cls):
        return get_config_dir_path() / "sd_lora_pokemon_1x8gb.yaml"

    #################
    # Event Handlers
    #################
    def on_reset_config_defaults_button_click(self):
        print("Resetting config defaults for SD LoRA.")
        self._current_config = self._default_config.model_copy(deep=True)
        return self.update_config_state(self._current_config)

    def on_generate_config_button_click(self, data: dict):
        print("Generating config for SD LoRA.")

        self._current_config.model = data.pop(self.model)
        self._current_config.hf_variant = data.pop(self.hf_variant)
        self._current_config.train_unet = data.pop(self.train_unet)
        self._current_config.unet_learning_rate = data.pop(self.unet_learning_rate)
        self._current_config.train_text_encoder = data.pop(self.train_text_encoder)
        self._current_config.text_encoder_learning_rate = data.pop(self.text_encoder_learning_rate)
        self._current_config.lr_scheduler = data.pop(self.lr_scheduler)
        self._current_config.lr_warmup_steps = data.pop(self.lr_warmup_steps)
        self._current_config.max_grad_norm = data.pop(self.max_grad_norm)
        self._current_config.train_batch_size = data.pop(self.train_batch_size)
        self._current_config.cache_text_encoder_outputs = data.pop(self.cache_text_encoder_outputs)
        self._current_config.cache_vae_outputs = data.pop(self.cache_vae_outputs)
        self._current_config.enable_cpu_offload_during_validation = data.pop(self.enable_cpu_offload_during_validation)
        self._current_config.gradient_accumulation_steps = data.pop(self.gradient_accumulation_steps)
        self._current_config.mixed_precision = data.pop(self.mixed_precision)
        self._current_config.gradient_checkpointing = data.pop(self.gradient_checkpointing)
        self._current_config.lora_rank_dim = data.pop(self.lora_rank_dim)
        self._current_config.num_validation_images_per_prompt = data.pop(self.num_validation_images_per_prompt)

        validation_prompts: list[str] = data.pop(self.validation_prompts).split("\n")
        validation_prompts = [x.strip() for x in validation_prompts if x.strip() != ""]
        self._current_config.validation_prompts = validation_prompts

        self.base_pipeline_config_group.update_config(self._current_config, data)
        self._current_config.optimizer = self.optimizer_config_group.update_config(data)

        # We pop items from data as we use them so that we can sanity check that all the input data was transferred to
        # the config.
        assert len(data) == 0

        # Roundtrip to make sure that the config is valid.
        self._current_config = SdLoraConfig.model_validate(self._current_config.model_dump())

        # Update the UI to reflect the new state of the config (in case some values were rounded or otherwise modified
        # in the process).
        update_dict = self.update_config_state(self._current_config)
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

    ########
    # Utils
    ########

    def get_all_configs(self):
        # HACK(ryand): This is a hack to avoid having to write a bunch of boilerplate code. We are assuming that all
        # public attributes are UI elements representing configs.
        all_attributes = vars(self)
        public_attributes = [all_attributes[k] for k in all_attributes.keys() if not k.startswith("_")]
        gradio_component_attributes = [x for x in public_attributes if isinstance(x, gr.components.Component)]

        return (
            gradio_component_attributes
            + self.base_pipeline_config_group.get_all_configs()
            + self.optimizer_config_group.get_all_configs()
        )

    def update_config_state(self, config: SdLoraConfig):
        unet_learning_rate = config.unet_learning_rate
        if unet_learning_rate is None:
            config.optimizer.learning_rate

        text_encoder_learning_rate = config.text_encoder_learning_rate
        if text_encoder_learning_rate is None:
            config.optimizer.learning_rate

        validation_prompts = "\n".join(config.validation_prompts)

        update_dict = {
            self.model: config.model,
            self.hf_variant: config.hf_variant,
            self.train_unet: config.train_unet,
            self.unet_learning_rate: unet_learning_rate,
            self.train_text_encoder: config.train_text_encoder,
            self.text_encoder_learning_rate: text_encoder_learning_rate,
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
            self.validation_prompts: validation_prompts,
            self.num_validation_images_per_prompt: config.num_validation_images_per_prompt,
        }
        update_dict.update(self.base_pipeline_config_group.update_config_state(config))
        update_dict.update(self.optimizer_config_group.update_config_state(config.optimizer))

        # Sanity check to prevent errors in all of the boilerplate code.
        assert set(update_dict.keys()) == set(self.get_all_configs())
        return update_dict

import gradio as gr
import yaml

from invoke_training.pipelines.stable_diffusion.lora.config import SdLoraConfig
from invoke_training.ui.base_pipeline_config_group import BasePipelineConfigGroup
from invoke_training.ui.optimizer_config_group import OptimizerConfigGroup
from invoke_training.ui.utils import get_config_dir_path, load_config_from_yaml


def get_default_config_file_path():
    return get_config_dir_path() / "sd_lora_pokemon_1x8gb.yaml"


class SdLoraTrainingTab:
    def __init__(self):
        default_config = load_config_from_yaml(get_default_config_file_path())
        assert isinstance(default_config, SdLoraConfig)
        self._default_config = default_config

        with gr.Tab(label="SD LoRA"):
            gr.Markdown("# SD LoRA Training Config")
            reset_config_defaults_button = gr.Button(value="Reset Config Defaults")

            gr.Markdown("## Basic Configs")
            self.model = gr.Textbox(label="model", info="TODO: Add more info", type="text", interactive=True)
            self.hf_variant = gr.Textbox(label="hf_variant", type="text", interactive=True)
            self.base_pipeline_config_group = BasePipelineConfigGroup()

            gr.Markdown("## Optimization")
            self.optimizer_config_group = OptimizerConfigGroup()
            with gr.Row():
                self.train_unet = gr.Checkbox(label="train_unet", interactive=True)
                self.unet_learning_rate = gr.Number(label="unet_learning_rate", interactive=True)
            with gr.Row():
                self.train_text_encoder = gr.Checkbox(label="train_text_encoder", interactive=True)
                self.text_encoder_learning_rate = gr.Number(label="text_encoder_learning_rate", interactive=True)

            gr.Markdown("## Config Output")
            generate_config_button = gr.Button(value="Generate Config")
            config_yaml = gr.Code(label="Config YAML", language="yaml", interactive=False)

        reset_config_defaults_button.click(
            self.on_reset_config_defaults_button_click, inputs=[], outputs=self.get_all_configs()
        )
        generate_config_button.click(
            self.on_generate_config_button_click, inputs=set(self.get_all_configs()), outputs=[config_yaml]
        )

    def on_reset_config_defaults_button_click(self):
        print("Resetting config defaults for SD LoRA.")
        return self.update_config_state(self._default_config)

    def on_generate_config_button_click(self, data):
        print("Generating config for SD LoRA.")
        # TODO: Don't start from the default config. Or, maybe we should but give users the ability to upload their own
        # default config.
        config = self._default_config.model_copy(deep=True)
        config.model = data[self.model]

        return yaml.safe_dump(config.model_dump(), default_flow_style=False, sort_keys=False)

    def get_all_configs(self):
        return (
            [
                self.model,
                self.hf_variant,
                self.train_unet,
                self.unet_learning_rate,
                self.train_text_encoder,
                self.text_encoder_learning_rate,
            ]
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

        update_dict = {
            self.model: config.model,
            self.hf_variant: config.hf_variant,
            self.train_unet: config.train_unet,
            self.unet_learning_rate: unet_learning_rate,
            self.train_text_encoder: config.train_text_encoder,
            self.text_encoder_learning_rate: text_encoder_learning_rate,
        }
        update_dict.update(self.base_pipeline_config_group.update_config_state(config))
        update_dict.update(self.optimizer_config_group.update_config_state(config.optimizer))
        return update_dict

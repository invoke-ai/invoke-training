import gradio as gr

from invoke_training.ui.sd_lora_tab import (
    SdLoraTrainingTab,
)


def launch_app():
    with gr.Blocks() as app:
        sd_lora_tab = SdLoraTrainingTab()

        # On app load, reset the configs for all tabs.
        app.load(sd_lora_tab.on_reset_config_defaults_button_click, inputs=[], outputs=sd_lora_tab.get_all_configs())

    app.launch()

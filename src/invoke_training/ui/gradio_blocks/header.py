import gradio as gr

from invoke_training.ui.utils.utils import get_assets_dir_path


class Header:
    def __init__(self):
        logo_path = get_assets_dir_path() / "logo.png"
        gr.Image(
            value=logo_path,
            label="Invoke Training App",
            width=200,
            interactive=False,
            container=False,
        )
        gr.Markdown(
            "[Home](/)\n\n"
            "*Invoke Training* - [Documentation](https://invoke-ai.github.io/invoke-training/) --"
            " Learn more about Invoke at [invoke.com](https://www.invoke.com/)"
        )

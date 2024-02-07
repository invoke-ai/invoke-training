import gradio as gr

from invoke_training.ui.utils import get_assets_dir_path


class DataPage:
    def __init__(self):
        logo_path = get_assets_dir_path() / "logo.png"
        with gr.Blocks(title="invoke-training", analytics_enabled=False) as app:
            with gr.Column():
                gr.Image(
                    value=logo_path,
                    label="Invoke Training App",
                    width=200,
                    interactive=False,
                    container=False,
                )
                with gr.Row():
                    gr.Markdown(
                        "*Invoke Training* - [Documentation](https://invoke-ai.github.io/invoke-training/) --"
                        " Learn more about Invoke at [invoke.com](https://www.invoke.com/)"
                    )

        self._app = app

    def app(self):
        return self._app

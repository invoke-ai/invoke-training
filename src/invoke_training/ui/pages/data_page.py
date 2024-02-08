from pathlib import Path

import gradio as gr

from invoke_training._shared.data.datasets.image_caption_jsonl_dataset import ImageCaptionJsonlDataset
from invoke_training._shared.utils.jsonl import save_jsonl
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

            gr.Markdown("# Data Annotation")

            gr.Markdown("To get started, either create a new dataset or load an existing one.")
            gr.Markdown(
                "Note: This UI creates datasets in `IMAGE_CAPTION_JSONL_DATASET` format. For more information about "
                "this format see [the docs](https://invoke-ai.github.io/invoke-training/concepts/dataset_formats/)"
            )

            gr.Markdown("## Setup")
            with gr.Group():
                # TODO: Expose image_column and caption_column as inputs?
                self._load_path_textbox = gr.Textbox(
                    label=".jsonl Path",
                    info="Enter the path to the .jsonl file to load or create.",
                    placeholder="/path/to/dataset.jsonl",
                )
                self._load_dataset_button = gr.Button("Load or Create Dataset")

            gr.Markdown("## Editing ")
            self._current_jsonl_textbox = gr.Textbox(
                label="Currently editing", interactive=False, placeholder="No dataset loaded"
            )
            self._current_len_number = gr.Number(label="Dataset length", interactive=False)

            self._load_dataset_button.click(
                self._on_load_dataset_button_click,
                inputs=set([self._load_path_textbox]),
                outputs=[self._current_jsonl_textbox, self._current_len_number],
            )

            self._app = app

    def _on_load_dataset_button_click(self, data: dict):
        jsonl_path = Path(data[self._load_path_textbox])
        jsonl_path = jsonl_path.resolve()
        if jsonl_path.exists():
            print(f"Loading dataset from '{jsonl_path}'.")
        else:
            print(f"Creating new dataset at '{jsonl_path}'.")
            assert jsonl_path.suffix == ".jsonl"
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            # Create an empty jsonl file.
            save_jsonl([], jsonl_path)

        # Initialize the dataset to validate the jsonl file, and to get the length.
        dataset = ImageCaptionJsonlDataset(jsonl_path)

        return {self._current_jsonl_textbox: jsonl_path, self._current_len_number: len(dataset)}

    def app(self):
        return self._app

from pathlib import Path

import gradio as gr

from invoke_training._shared.data.datasets.image_caption_jsonl_dataset import (
    CAPTION_COLUMN_DEFAULT,
    IMAGE_COLUMN_DEFAULT,
    ImageCaptionJsonlDataset,
)
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
                self._jsonl_path_textbox = gr.Textbox(
                    label=".jsonl Path",
                    info="Enter the path to the .jsonl file to load or create.",
                    placeholder="/path/to/dataset.jsonl",
                )
                with gr.Row():
                    self._image_column_textbox = gr.Textbox(
                        label="Image Column (Optional)", placeholder=IMAGE_COLUMN_DEFAULT
                    )
                    self._caption_column_textbox = gr.Textbox(
                        label="Caption Column (Optional)", placeholder=CAPTION_COLUMN_DEFAULT
                    )
                self._load_dataset_button = gr.Button("Load or Create Dataset")

            gr.Markdown("## Edit Captions")
            self._cur_len_number = gr.Number(label="Dataset length", interactive=False)

            self._cur_example_index = gr.Number(label="Current index", precision=0, interactive=False)
            self._cur_image = gr.Image(value=None, label="Image", interactive=False, width=500)
            self._cur_caption = gr.Textbox(label="Caption", interactive=True)
            with gr.Row():
                self._save_and_prev_button = gr.Button("Save and Go-To Previous")
                self._save_and_next_button = gr.Button("Save and Go-To Next")

            self._app = app

            self._load_dataset_button.click(
                self._on_load_dataset_button_click,
                inputs=set([self._jsonl_path_textbox, self._image_column_textbox, self._caption_column_textbox]),
                outputs=[
                    self._cur_len_number,
                    self._cur_example_index,
                    self._cur_image,
                    self._cur_caption,
                ],
            )
            self._save_and_prev_button.click(
                self._on_save_and_prev_button_click,
                inputs=set(
                    [
                        self._jsonl_path_textbox,
                        self._image_column_textbox,
                        self._caption_column_textbox,
                        self._cur_example_index,
                        self._cur_caption,
                    ]
                ),
                outputs=[
                    self._cur_len_number,
                    self._cur_example_index,
                    self._cur_image,
                    self._cur_caption,
                ],
            )
            self._save_and_next_button.click(
                self._on_save_and_next_button_click,
                inputs=set(
                    [
                        self._jsonl_path_textbox,
                        self._image_column_textbox,
                        self._caption_column_textbox,
                        self._cur_example_index,
                        self._cur_caption,
                    ]
                ),
                outputs=[
                    self._cur_len_number,
                    self._cur_example_index,
                    self._cur_image,
                    self._cur_caption,
                ],
            )

    def _update_state(self, dataset: ImageCaptionJsonlDataset, idx: int):
        idx = idx
        image = None
        caption = None
        if 0 <= idx and idx < len(dataset):
            example = dataset[idx]
            image = example["image"]
            caption = example["caption"]

        return {
            self._cur_len_number: len(dataset),
            self._cur_example_index: idx,
            self._cur_image: image,
            self._cur_caption: caption,
        }

    def _on_load_dataset_button_click(self, data: dict):
        jsonl_path = Path(data[self._jsonl_path_textbox])
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
        dataset = ImageCaptionJsonlDataset(
            jsonl_path=jsonl_path,
            image_column=data[self._image_column_textbox] or IMAGE_COLUMN_DEFAULT,
            caption_column=data[self._caption_column_textbox] or CAPTION_COLUMN_DEFAULT,
        )

        return self._update_state(dataset, 0)

    def _on_save_and_go_button_click(self, data: dict, idx_change: int):
        jsonl_path = Path(data[self._jsonl_path_textbox])
        dataset = ImageCaptionJsonlDataset(
            jsonl_path=jsonl_path,
            image_column=data[self._image_column_textbox] or IMAGE_COLUMN_DEFAULT,
            caption_column=data[self._caption_column_textbox] or CAPTION_COLUMN_DEFAULT,
        )

        # Update the current caption and re-save the jsonl file.
        idx: int = data[self._cur_example_index]
        print(f"Updating caption for example {idx} of '{jsonl_path}'.")
        caption = data[self._cur_caption]
        dataset.examples[idx].caption = caption
        dataset.save_jsonl()

        return self._update_state(dataset, idx + idx_change)

    def _on_save_and_next_button_click(self, data: dict):
        return self._on_save_and_go_button_click(data, 1)

    def _on_save_and_prev_button_click(self, data: dict):
        return self._on_save_and_go_button_click(data, -1)

    def app(self):
        return self._app

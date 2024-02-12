from pathlib import Path

import gradio as gr

from invoke_training._shared.data.datasets.image_caption_jsonl_dataset import (
    CAPTION_COLUMN_DEFAULT,
    IMAGE_COLUMN_DEFAULT,
    ImageCaptionExample,
    ImageCaptionJsonlDataset,
)
from invoke_training._shared.utils.jsonl import save_jsonl
from invoke_training.ui.gradio_blocks.header import Header

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


class DataPage:
    def __init__(self):
        # The dataset that is currently being edited.
        self._jsonl_path: str | None = None
        self._dataset: ImageCaptionJsonlDataset | None = None

        with gr.Blocks(
            title="invoke-training",
            analytics_enabled=False,
            head='<link rel="icon" type="image/x-icon" href="/assets/favicon.png">',
        ) as app:
            self._header = Header()
            gr.Markdown("# Data Annotation")
            gr.Markdown(
                "Note: This UI creates datasets in `IMAGE_CAPTION_JSONL_DATASET` format. For more information about "
                "this format see [the docs](https://invoke-ai.github.io/invoke-training/concepts/dataset_formats/)"
            )

            # HACK: I use a column as a wrapper to control visbility of this group of UI elements. gr.Group sounds like
            # a more natural choice for this purpose, but it applies some styling that makes the group look weird.
            with gr.Column() as select_dataset_group:
                gr.Markdown("## Load Existing Dataset")
                with gr.Group():
                    self._existing_jsonl_path = gr.Textbox(
                        label="Existing .jsonl Path",
                        info="Enter the path to an existing dataset's .jsonl file.",
                        placeholder="/path/to/dataset.jsonl",
                    )
                    with gr.Row():
                        self._image_column_textbox = gr.Textbox(
                            label="Image Column (Optional)", placeholder=IMAGE_COLUMN_DEFAULT
                        )
                        self._caption_column_textbox = gr.Textbox(
                            label="Caption Column (Optional)", placeholder=CAPTION_COLUMN_DEFAULT
                        )
                    self._load_existing_dataset_button = gr.Button("Load Existing Dataset")
                gr.Markdown("## Create New Dataset")
                with gr.Group():
                    self._new_jsonl_path = gr.Textbox(
                        label="New .jsonl Path",
                        info="Enter the path for a new .jsonl file.",
                        placeholder="/path/to/dataset.jsonl",
                    )
                    self._create_new_dataset_button = gr.Button("Create New Dataset")
            self._select_dataset_group = select_dataset_group

            # HACK: I use a column as a wrapper to control visbility of this group of UI elements. gr.Group sounds like
            # a more natural choice for this purpose, but it applies some styling that makes the group look weird.
            with gr.Column(visible=False) as edit_dataset_group:
                with gr.Row():
                    self._current_jsonl_path = gr.Textbox(label="Currently editing:", interactive=False)
                    self._change_dataset_button = gr.Button("Change")
                gr.Markdown("## Add Images")
                with gr.Group():
                    self._image_source_textbox = gr.Textbox(
                        label="Image Source",
                        info="Enter the path to a single image or a directory containing images. If a directory path "
                        "is passed, it will be searched recursively for image files.",
                        placeholder="/path/to/image_dir",
                    )
                    self._add_images_button = gr.Button("Add Images")

                gr.Markdown("## Edit Captions")
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            self._cur_example_index = gr.Number(label="Current index", precision=0, interactive=True)
                            self._cur_len_number = gr.Number(label="Dataset length", interactive=False)
                        with gr.Row():
                            self._beyond_dataset_limits_warning = gr.Markdown("**Current index is beyond dataset limits.** If you have completed all captions, click 'Home' to begin training.")
                        with gr.Row():
                            self._cur_image = gr.Image(value=None, label="Image", interactive=False, width=500)
                    with gr.Column():
                            self._cur_caption = gr.Textbox(label="Caption", interactive=True, lines=25)


                with gr.Row():
                    self._save_and_prev_button = gr.Button("Save and Go-To Previous")
                    self._save_and_next_button = gr.Button("Save and Go-To Next")

                gr.Markdown("## Raw JSONL")
                self._data_jsonl = gr.Code(label="Dataset .jsonl", language="json", interactive=False)

            self._edit_dataset_group = edit_dataset_group
            self._app = app

            standard_outputs = [
                self._select_dataset_group,
                self._edit_dataset_group,
                self._current_jsonl_path,
                self._cur_len_number,
                self._cur_example_index,
                self._cur_image,
                self._cur_caption,
                self._beyond_dataset_limits_warning,
                self._data_jsonl,
            ]

            self._load_existing_dataset_button.click(
                self._on_load_existing_dataset_button_click,
                inputs=set([self._existing_jsonl_path, self._image_column_textbox, self._caption_column_textbox]),
                outputs=standard_outputs,
            )

            self._create_new_dataset_button.click(
                self._on_create_dataset_button_click,
                inputs=set([self._new_jsonl_path]),
                outputs=standard_outputs,
            )

            self._change_dataset_button.click(
                self._on_change_dataset_button_click, inputs=None, outputs=standard_outputs
            )
            self._save_and_prev_button.click(
                self._on_save_and_prev_button_click,
                inputs=set([self._cur_example_index, self._cur_caption]),
                outputs=standard_outputs,
            )

            self._save_and_next_button.click(
                self._on_save_and_next_button_click,
                inputs=set([self._cur_example_index, self._cur_caption]),
                outputs=standard_outputs,
            )

            self._add_images_button.click(
                self._on_add_images_button_click,
                inputs=set([self._image_source_textbox]),
                outputs=standard_outputs,
            )

            self._cur_example_index.input(
                self._on_cur_example_index_change,
                inputs=set([self._cur_example_index]),
                outputs=standard_outputs,
            )

    def _update_state(self, idx: int):
        if self._dataset is None or self._jsonl_path is None:
            return {
                self._select_dataset_group: gr.Group(visible=True),
                self._edit_dataset_group: gr.Column(visible=False),
                self._current_jsonl_path: None,
                self._cur_len_number: 0,
                self._cur_example_index: 0,
                self._cur_image: None,
                self._cur_caption: None,
                self._beyond_dataset_limits_warning: gr.Markdown(visible=False),
                self._data_jsonl: "",
            }

        idx = idx
        image = None
        caption = None
        beyond_limits = True
        if 0 <= idx and idx < len(self._dataset):
            beyond_limits = False
            example = self._dataset[idx]
            image = example["image"]
            caption = example["caption"]

        jsonl_str = "\n".join([example.model_dump_json() for example in self._dataset.examples])
        return {
            self._select_dataset_group: gr.Group(visible=self._dataset is None),
            self._edit_dataset_group: gr.Column(visible=self._dataset is not None),
            self._current_jsonl_path: str(self._jsonl_path),
            self._cur_len_number: len(self._dataset),
            self._cur_example_index: idx,
            self._cur_image: image,
            self._cur_caption: caption,
            self._beyond_dataset_limits_warning: gr.Markdown(visible=beyond_limits),
            self._data_jsonl: jsonl_str,
        }

    def _on_load_existing_dataset_button_click(self, data: dict):
        """Load an existing dataset."""
        jsonl_path = Path(data[self._existing_jsonl_path])
        jsonl_path = jsonl_path.resolve()
        if not jsonl_path.exists():
            raise ValueError(f"'{jsonl_path}' does not exist.")

        self._jsonl_path = jsonl_path
        self._dataset = ImageCaptionJsonlDataset(
            jsonl_path=jsonl_path,
            image_column=data[self._image_column_textbox] or IMAGE_COLUMN_DEFAULT,
            caption_column=data[self._caption_column_textbox] or CAPTION_COLUMN_DEFAULT,
        )
        return self._update_state(0)

    def _on_create_dataset_button_click(self, data: dict):
        """Create a new dataset."""
        jsonl_path = Path(data[self._new_jsonl_path])
        jsonl_path = jsonl_path.resolve()
        if jsonl_path.exists():
            raise ValueError(f"'{jsonl_path}' already exists.")

        if jsonl_path.suffix != ".jsonl":
            raise ValueError("Invalid file extension. Expected '.jsonl'.")

        print(f"Creating new dataset at '{jsonl_path}'.")
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        # Create an empty jsonl file.
        save_jsonl([], jsonl_path)

        self._jsonl_path = jsonl_path
        self._dataset = ImageCaptionJsonlDataset(jsonl_path=jsonl_path)

        return self._update_state(0)

    def _on_change_dataset_button_click(self):
        self._jsonl_path = None
        self._dataset = None
        return self._update_state(0)

    def _on_save_and_go_button_click(self, data: dict, idx_change: int):
        # Update the current caption and re-save the jsonl file.
        idx: int = data[self._cur_example_index]
        if idx < 0 or idx >= len(self._dataset):
            # idx is out of bounds, so don't update the caption, but still change the index.
            return self._update_state(idx + idx_change)

        print(f"Updating caption for example {idx} of '{self._jsonl_path}'.")
        caption = data[self._cur_caption]
        self._dataset.examples[idx].caption = caption
        self._dataset.save_jsonl()

        return self._update_state(idx + idx_change)

    def _on_save_and_next_button_click(self, data: dict):
        return self._on_save_and_go_button_click(data, 1)

    def _on_save_and_prev_button_click(self, data: dict):
        return self._on_save_and_go_button_click(data, -1)

    def _on_cur_example_index_change(self, data: dict):
        return self._update_state(data[self._cur_example_index])

    def _on_add_images_button_click(self, data: dict):
        """Add images to the dataset."""
        image_source_path = Path(data[self._image_source_textbox])

        if not image_source_path.exists():
            raise ValueError(f"'{image_source_path}' does not exist.")

        # Determine the list of image paths to add to the dataset.
        image_paths = []
        if image_source_path.is_file():
            if image_source_path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(
                    f"'{image_source_path}' is not a valid image file. Expected one of {IMAGE_EXTENSIONS}."
                )

            image_paths.append(image_source_path.resolve())
        else:
            # Recursively search for image files in the image_source_path directory.
            for file_path in image_source_path.glob("**/*"):
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(file_path.resolve())

        # Avoid adding duplicate images.
        cur_image_paths = set([Path(example.image_path) for example in self._dataset.examples])
        image_paths = set(image_paths)
        new_image_paths = image_paths - cur_image_paths
        if len(new_image_paths) < len(image_paths):
            print(f"Skipping {len(image_paths) - len(new_image_paths)} images that are already in the dataset.")

        # Add the new images to the dataset.
        print(f"Adding {len(new_image_paths)} images to '{self._jsonl_path}'.")
        for image_path in new_image_paths:
            self._dataset.examples.append(ImageCaptionExample(image_path=str(image_path), caption=""))

        # Save the updated dataset.
        self._dataset.save_jsonl()

        return self._update_state(0)

    def app(self):
        return self._app

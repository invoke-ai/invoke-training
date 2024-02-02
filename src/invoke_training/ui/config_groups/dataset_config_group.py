from typing import Any

import gradio as gr

from invoke_training.config.data.dataset_config import (
    HFDirImageCaptionDatasetConfig,
    HFHubImageCaptionDatasetConfig,
    ImageCaptionDatasetConfig,
    ImageDirDatasetConfig,
)
from invoke_training.ui.config_groups.ui_config_element import UIConfigElement

ALL_DATASET_TYPES = ["HF_HUB_IMAGE_CAPTION_DATASET", "HF_DIR_IMAGE_CAPTION_DATASET", "IMAGE_DIR_DATASET"]


class HFHubImageCaptionDatasetConfigGroup(UIConfigElement):
    def __init__(self):
        self.dataset_name = gr.Textbox(label="dataset_name", interactive=True)
        with gr.Row():
            self.dataset_config_name = gr.Textbox(label="dataset_config_name", interactive=True)
            self.hf_cache_dir = gr.Textbox(label="hf_cache_dir", interactive=True)
        # self.image_column = gr.Textbox(label="image_column", interactive=True)
        # self.caption_column = gr.Textbox(label="caption_column", interactive=True)

    def update_ui_components_with_config_data(
        self, config: HFHubImageCaptionDatasetConfig | None
    ) -> dict[gr.components.Component, Any]:
        return {
            self.dataset_name: config.dataset_name if config else "<insert_dataset_name>",
            self.dataset_config_name: config.dataset_config_name if config else None,
            self.hf_cache_dir: config.hf_cache_dir if config else None,
            # self.image_column: config.image_column,
            # self.caption_column: config.caption_column,
        }

    def update_config_with_ui_component_data(
        self, orig_config: HFHubImageCaptionDatasetConfig | None, ui_data: dict[gr.components.Component, Any]
    ) -> HFHubImageCaptionDatasetConfig:
        assert orig_config is None
        # new_config = orig_config.model_copy(deep=True)

        new_config = HFHubImageCaptionDatasetConfig(
            dataset_name=ui_data.pop(self.dataset_name),
            dataset_config_name=ui_data.pop(self.dataset_config_name) or None,
            hf_cache_dir=ui_data.pop(self.hf_cache_dir) or None,
            # image_column=ui_data.pop(self.image_column),
            # caption_column=ui_data.pop(self.caption_column),
        )
        return new_config


class HFDirImageCaptionDatasetConfigGroup(UIConfigElement):
    def __init__(self):
        self.dataset_dir = gr.Textbox(label="dataset_dir", interactive=True)
        # self.image_column = gr.Textbox(label="image_column", interactive=True)
        # self.caption_column = gr.Textbox(label="caption_column", interactive=True)

    def update_ui_components_with_config_data(
        self, config: HFDirImageCaptionDatasetConfig | None
    ) -> dict[gr.components.Component, Any]:
        return {
            self.dataset_dir: config.dataset_dir if config else "<path/to/dataset_dir>",
            # self.image_column: config.image_column,
            # self.caption_column: config.caption_column,
        }

    def update_config_with_ui_component_data(
        self, orig_config: HFDirImageCaptionDatasetConfig | None, ui_data: dict[gr.components.Component, Any]
    ) -> HFDirImageCaptionDatasetConfig:
        assert orig_config is None
        # new_config = orig_config.model_copy(deep=True)

        new_config = HFDirImageCaptionDatasetConfig(
            dataset_dir=ui_data.pop(self.dataset_dir),
            # image_column=ui_data.pop(self.image_column),
            # caption_column=ui_data.pop(self.caption_column),
        )
        return new_config


class ImageDirDatasetConfigGroup(UIConfigElement):
    def __init__(self):
        with gr.Row():
            self.dataset_dir = gr.Textbox(label="dataset_dir", interactive=True)
            self.keep_in_memory = gr.Checkbox(label="keep_in_memory", interactive=True)

    def update_ui_components_with_config_data(
        self, config: ImageDirDatasetConfig | None
    ) -> dict[gr.components.Component, Any]:
        return {
            self.dataset_dir: config.dataset_dir if config else "<path/to/dataset_dir>",
            self.keep_in_memory: config.keep_in_memory if config else False,
        }

    def update_config_with_ui_component_data(
        self, orig_config: ImageDirDatasetConfig | None, ui_data: dict[gr.components.Component, Any]
    ) -> ImageDirDatasetConfig:
        assert orig_config is None
        # new_config = orig_config.model_copy(deep=True)

        new_config = ImageDirDatasetConfig(
            dataset_dir=ui_data.pop(self.dataset_dir), keep_in_memory=ui_data.pop(self.keep_in_memory)
        )
        return new_config


class DatasetConfigGroup(UIConfigElement):
    def __init__(self, allowed_types: list[str]):
        self.type = gr.Dropdown(
            choices=[t for t in ALL_DATASET_TYPES if t in allowed_types], label="type", interactive=True
        )

        with gr.Group() as hf_hub_image_caption_dataset_config_group:
            self.hf_hub_image_caption_dataset_config = HFHubImageCaptionDatasetConfigGroup()
        self.hf_hub_image_caption_dataset_config_group = hf_hub_image_caption_dataset_config_group

        with gr.Group() as hf_dir_image_caption_dataset_config_group:
            self.hf_dir_image_caption_dataset_config = HFDirImageCaptionDatasetConfigGroup()
        self.hf_dir_image_caption_dataset_config_group = hf_dir_image_caption_dataset_config_group

        with gr.Group() as image_dir_dataset_config_group:
            self.image_dir_dataset_config = ImageDirDatasetConfigGroup()
        self.image_dir_dataset_config_group = image_dir_dataset_config_group

        self.type.change(
            self._on_type_change,
            inputs=[self.type],
            outputs=[
                self.hf_hub_image_caption_dataset_config_group,
                self.hf_dir_image_caption_dataset_config_group,
                self.image_dir_dataset_config_group,
            ],
        )

    def _on_type_change(self, type: str):
        return {
            self.hf_hub_image_caption_dataset_config_group: gr.Group(visible=type == "HF_HUB_IMAGE_CAPTION_DATASET"),
            self.hf_dir_image_caption_dataset_config_group: gr.Group(visible=type == "HF_DIR_IMAGE_CAPTION_DATASET"),
            self.image_dir_dataset_config_group: gr.Group(visible=type == "IMAGE_DIR_DATASET"),
        }

    def update_ui_components_with_config_data(
        self, config: ImageCaptionDatasetConfig
    ) -> dict[gr.components.Component, Any]:
        update_dict = {
            self.type: config.type,
            self.hf_hub_image_caption_dataset_config_group: gr.Group(
                visible=config.type == "HF_HUB_IMAGE_CAPTION_DATASET"
            ),
            self.hf_dir_image_caption_dataset_config_group: gr.Group(
                visible=config.type == "HF_DIR_IMAGE_CAPTION_DATASET"
            ),
            self.image_dir_dataset_config_group: gr.Group(visible=config.type == "IMAGE_DIR_DATASET"),
        }

        update_dict.update(
            self.hf_hub_image_caption_dataset_config.update_ui_components_with_config_data(
                config if config.type == "HF_HUB_IMAGE_CAPTION_DATASET" else None
            )
        )
        update_dict.update(
            self.hf_dir_image_caption_dataset_config.update_ui_components_with_config_data(
                config if config.type == "HF_DIR_IMAGE_CAPTION_DATASET" else None
            )
        )
        update_dict.update(
            self.image_dir_dataset_config.update_ui_components_with_config_data(
                config if config.type == "IMAGE_DIR_DATASET" else None
            )
        )

        return update_dict

    def update_config_with_ui_component_data(
        self, orig_config: ImageCaptionDatasetConfig, ui_data: dict[gr.components.Component, Any]
    ) -> ImageCaptionDatasetConfig:
        # TODO: Use orig_config.

        new_config_hf_hub = self.hf_hub_image_caption_dataset_config.update_config_with_ui_component_data(None, ui_data)
        new_config_hf_dir = self.hf_dir_image_caption_dataset_config.update_config_with_ui_component_data(None, ui_data)
        new_config_image_dir = self.image_dir_dataset_config.update_config_with_ui_component_data(None, ui_data)

        type = ui_data.pop(self.type)
        if type == "HF_HUB_IMAGE_CAPTION_DATASET":
            new_config = new_config_hf_hub
        elif type == "HF_DIR_IMAGE_CAPTION_DATASET":
            new_config = new_config_hf_dir
        elif type == "IMAGE_DIR_DATASET":
            new_config = new_config_image_dir
        else:
            raise ValueError(f"Unknown dataset type: {type}")

        return new_config

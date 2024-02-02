from typing import Any

import gradio as gr

from invoke_training.config.data.data_loader_config import ImageCaptionSDDataLoaderConfig
from invoke_training.ui.config_groups.dataset_config_group import DatasetConfigGroup
from invoke_training.ui.config_groups.ui_config_element import UIConfigElement


class ImageCaptionSDDataLoaderConfigGroup(UIConfigElement):
    # TODO: Add aspect_ratio_buckets
    def __init__(self):
        self.dataset = DatasetConfigGroup(
            allowed_types=["HF_HUB_IMAGE_CAPTION_DATASET", "HF_DIR_IMAGE_CAPTION_DATASET"]
        )
        self.resolution = gr.Number(label="resolution", precision=0, interactive=True)
        self.center_crop = gr.Checkbox(label="center_crop", interactive=True)
        self.random_flip = gr.Checkbox(label="random_flip", interactive=True)
        self.caption_prefix = gr.Textbox(label="caption_prefix", interactive=True)
        self.dataloader_num_workers = gr.Number(label="dataloader_num_workers", precision=0, interactive=True)

    def update_ui_components_with_config_data(
        self, config: ImageCaptionSDDataLoaderConfig
    ) -> dict[gr.components.Component, Any]:
        update_dict = {
            self.resolution: config.resolution,
            self.center_crop: config.center_crop,
            self.random_flip: config.random_flip,
            self.caption_prefix: config.caption_prefix,
            self.dataloader_num_workers: config.dataloader_num_workers,
        }

        update_dict.update(self.dataset.update_ui_components_with_config_data(config.dataset))

        return update_dict

    def update_config_with_ui_component_data(
        self, orig_config: ImageCaptionSDDataLoaderConfig, ui_data: dict[gr.components.Component, Any]
    ) -> ImageCaptionSDDataLoaderConfig:
        new_config = orig_config.model_copy(deep=True)

        new_config.dataset = self.dataset.update_config_with_ui_component_data(orig_config.dataset, ui_data)
        new_config.resolution = ui_data.pop(self.resolution)
        new_config.center_crop = ui_data.pop(self.center_crop)
        new_config.random_flip = ui_data.pop(self.random_flip)
        new_config.caption_prefix = ui_data.pop(self.caption_prefix) or None
        new_config.dataloader_num_workers = ui_data.pop(self.dataloader_num_workers)

        return new_config

from typing import Any

import gradio as gr

from invoke_training.config.data.data_loader_config import ImageCaptionSDDataLoaderConfig
from invoke_training.ui.config_groups.aspect_ratio_bucket_config_group import AspectRatioBucketConfigGroup
from invoke_training.ui.config_groups.dataset_config_group import DatasetConfigGroup
from invoke_training.ui.config_groups.ui_config_element import UIConfigElement


class ImageCaptionSDDataLoaderConfigGroup(UIConfigElement):
    def __init__(self):
        with gr.Tab("Data Source Configs"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        self.dataset = DatasetConfigGroup(
                            allowed_types=[
                                "HF_HUB_IMAGE_CAPTION_DATASET",
                                "IMAGE_CAPTION_JSONL_DATASET",
                                "IMAGE_CAPTION_DIR_DATASET",
                            ]
                        )
                with gr.Column(scale=3):
                    with gr.Tab("Data Loading Configs"):
                        with gr.Group():
                            with gr.Row():
                                self.resolution = gr.Number(
                                    label="Resolution",
                                    info="The resolution for input images. All of the images in the dataset will be"
                                    " resized to this resolution unless the aspect_ratio_buckets config is set.",
                                    precision=0,
                                    interactive=True,
                                )
                                self.dataloader_num_workers = gr.Number(
                                    label="Dataloading Workers",
                                    info="Number of subprocesses to use for data loading. 0 means that the data will"
                                    " be loaded in the main process.",
                                    precision=0,
                                    interactive=True,
                                )
                            with gr.Row():
                                self.center_crop = gr.Checkbox(
                                    label="Center Crop",
                                    info="If set, input images will be center-cropped to the target resolution."
                                    " Otherwise, input images will be randomly cropped to the target resolution.",
                                    interactive=True,
                                )
                                self.random_flip = gr.Checkbox(
                                    label="Random Flip",
                                    info="If set, random flip augmentations will be applied to input images.",
                                    interactive=True,
                                )
                            self.caption_prefix = gr.Textbox(
                                label="Caption Prefix",
                                info="A prefix that will be prepended to all captions."
                                " If None, no prefix will be added.",
                                interactive=True,
                            )
                    with gr.Tab("Aspect Ratio Bucketing Configs"):
                        self.aspect_ratio_bucket_config_group = AspectRatioBucketConfigGroup()

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
        update_dict.update(
            self.aspect_ratio_bucket_config_group.update_ui_components_with_config_data(config.aspect_ratio_buckets)
        )

        return update_dict

    def update_config_with_ui_component_data(
        self, orig_config: ImageCaptionSDDataLoaderConfig, ui_data: dict[gr.components.Component, Any]
    ) -> ImageCaptionSDDataLoaderConfig:
        new_config = orig_config.model_copy(deep=True)

        new_config.dataset = self.dataset.update_config_with_ui_component_data(orig_config.dataset, ui_data)
        new_config.aspect_ratio_buckets = self.aspect_ratio_bucket_config_group.update_config_with_ui_component_data(
            orig_config.aspect_ratio_buckets, ui_data
        )
        new_config.resolution = ui_data.pop(self.resolution)
        new_config.center_crop = ui_data.pop(self.center_crop)
        new_config.random_flip = ui_data.pop(self.random_flip)
        new_config.caption_prefix = ui_data.pop(self.caption_prefix) or None
        new_config.dataloader_num_workers = ui_data.pop(self.dataloader_num_workers)

        return new_config

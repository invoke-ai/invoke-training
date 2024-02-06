from typing import Any

import gradio as gr

from invoke_training.config.data.data_loader_config import (
    TextualInversionSDDataLoaderConfig,
)
from invoke_training.ui.config_groups.aspect_ratio_bucket_config_group import AspectRatioBucketConfigGroup
from invoke_training.ui.config_groups.dataset_config_group import DatasetConfigGroup
from invoke_training.ui.config_groups.ui_config_element import UIConfigElement


class TextualInversionSDDataLoaderConfigGroup(UIConfigElement):
    def __init__(self):
        with gr.Tab("Data Source Configs"):
            with gr.Group():
                self.dataset = DatasetConfigGroup(
                    allowed_types=[
                        "HF_HUB_IMAGE_CAPTION_DATASET",
                        "IMAGE_CAPTION_JSONL_DATASET",
                        "IMAGE_CAPTION_DIR_DATASET",
                        "IMAGE_DIR_DATASET",
                    ]
                )

        with gr.Tab("Data Loading Configs"):
            with gr.Group():
                self.caption_preset = gr.Dropdown(
                    label="Caption Preset",
                    choices=["None", "style", "object"],
                    info="Only one of 'Caption Preset' or 'Caption Templates' should be set.\nSelect a Caption Preset "
                    "option to use a set of pre-configured templates.",
                    interactive=True,
                )
                self.caption_templates = gr.Textbox(
                    label="Caption Templates",
                    info="Only one of 'Caption Preset' or 'Caption Templates' should be set. Enter one template per "
                    "line. Each template should contain a single placeholder token slot indicated by '{}', for example "
                    "'a photo of a {}'.",
                    lines=5,
                    interactive=True,
                )
                with gr.Row():
                    self.keep_original_captions = gr.Checkbox(
                        label="Keep Original Captions",
                        info="If True, the caption templates will be prepended to the original dataset captions. If "
                        "False, the caption templates will replace the original captions.",
                        interactive=True,
                    )
                    self.shuffle_caption_delimiter = gr.Textbox(
                        label="Shuffle Caption Delimiter",
                        info="If set, captions will be split on the provided delimiter (e.g. ',') and shuffled.",
                        interactive=True,
                    )

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
                        info="If set, input images will be center-cropped to the target resolution. Otherwise,"
                        " input images will be randomly cropped to the target resolution.",
                        interactive=True,
                    )
                    self.random_flip = gr.Checkbox(
                        label="Random Flip",
                        info="If set, random flip augmentations will be applied to input images.",
                        interactive=True,
                    )
        with gr.Tab("Aspect Ratio Bucketing Configs"):
            self.aspect_ratio_bucket_config_group = AspectRatioBucketConfigGroup()

    def update_ui_components_with_config_data(
        self, config: TextualInversionSDDataLoaderConfig
    ) -> dict[gr.components.Component, Any]:
        # Special handling of caption_preset to translate None to "None".
        caption_preset = "None"
        if config.caption_preset is not None:
            caption_preset = config.caption_preset

        update_dict = {
            self.caption_preset: caption_preset,
            self.caption_templates: "\n".join(config.caption_templates or []),
            self.keep_original_captions: config.keep_original_captions,
            self.shuffle_caption_delimiter: config.shuffle_caption_delimiter,
            self.resolution: config.resolution,
            self.center_crop: config.center_crop,
            self.random_flip: config.random_flip,
            self.dataloader_num_workers: config.dataloader_num_workers,
        }

        update_dict.update(self.dataset.update_ui_components_with_config_data(config.dataset))
        update_dict.update(
            self.aspect_ratio_bucket_config_group.update_ui_components_with_config_data(config.aspect_ratio_buckets)
        )

        return update_dict

    def update_config_with_ui_component_data(
        self, orig_config: TextualInversionSDDataLoaderConfig, ui_data: dict[gr.components.Component, Any]
    ) -> TextualInversionSDDataLoaderConfig:
        new_config = orig_config.model_copy(deep=True)

        # Special handling of caption_preset to translate "None" to None.
        caption_presets = {"None": None, "style": "style", "object": "object"}
        caption_preset = caption_presets[ui_data.pop(self.caption_preset)]

        # Special handling of caption_templates.
        caption_templates: list[str] = ui_data.pop(self.caption_templates).split("\n")
        caption_templates = [x.strip() for x in caption_templates if x.strip() != ""] or None

        new_config.dataset = self.dataset.update_config_with_ui_component_data(orig_config.dataset, ui_data)
        new_config.aspect_ratio_buckets = self.aspect_ratio_bucket_config_group.update_config_with_ui_component_data(
            orig_config.aspect_ratio_buckets, ui_data
        )
        new_config.caption_preset = caption_preset
        new_config.caption_templates = caption_templates
        new_config.keep_original_captions = ui_data.pop(self.keep_original_captions)
        new_config.shuffle_caption_delimiter = ui_data.pop(self.shuffle_caption_delimiter) or None
        new_config.resolution = ui_data.pop(self.resolution)
        new_config.center_crop = ui_data.pop(self.center_crop)
        new_config.random_flip = ui_data.pop(self.random_flip)
        new_config.dataloader_num_workers = ui_data.pop(self.dataloader_num_workers)

        return new_config

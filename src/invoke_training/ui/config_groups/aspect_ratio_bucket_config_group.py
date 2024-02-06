from typing import Any

import gradio as gr

from invoke_training.config.data.data_loader_config import AspectRatioBucketConfig
from invoke_training.ui.config_groups.ui_config_element import UIConfigElement


class AspectRatioBucketConfigGroup(UIConfigElement):
    def __init__(self):
        gr.Markdown(
            "Aspect ratio bucket resolutions are generated as follows:\n"
            "- Iterate over 'first' dimension values from `start_dim` to `end_dim` in steps of size `divisible_by`.\n"
            "- Calculate the 'second' dimension to be as close as possible to the total number of pixels in "
            "`target_resolution`, while still being divisible by `divisible_by`."
        )
        self.enabled = gr.Checkbox(label="Use Aspect Ratio Bucketing", interactive=True)
        self.target_resolution = gr.Number(label="target_resolution", interactive=True, precision=0)
        self.start_dim = gr.Number(label="start_dimension", interactive=True, precision=0)
        self.end_dim = gr.Number(label="end_imension", interactive=True, precision=0)
        self.divisible_by = gr.Number(label="divisible_by", interactive=True, precision=0)

    def update_ui_components_with_config_data(
        self, config: AspectRatioBucketConfig | None
    ) -> dict[gr.components.Component, Any]:
        enabled = True
        if config is None:
            enabled = False
            # We just construct this config to hold default values.
            config = AspectRatioBucketConfig(target_resolution=512, start_dim=256, end_dim=768, divisible_by=64)

        update_dict = {
            self.enabled: enabled,
            self.target_resolution: config.target_resolution,
            self.start_dim: config.start_dim,
            self.end_dim: config.end_dim,
            self.divisible_by: config.divisible_by,
        }
        return update_dict

    def update_config_with_ui_component_data(
        self, orig_config: AspectRatioBucketConfig | None, ui_data: dict[gr.components.Component, Any]
    ) -> AspectRatioBucketConfig | None:
        # TODO: Use orig_config?
        if not ui_data.pop(self.enabled):
            # Pop fields from ui_data so that upstream code knows that the fields were handled.
            ui_data.pop(self.target_resolution)
            ui_data.pop(self.start_dim)
            ui_data.pop(self.end_dim)
            ui_data.pop(self.divisible_by)
            return None

        new_config = AspectRatioBucketConfig(
            target_resolution=ui_data.pop(self.target_resolution),
            start_dim=ui_data.pop(self.start_dim),
            end_dim=ui_data.pop(self.end_dim),
            divisible_by=ui_data.pop(self.divisible_by),
        )
        return new_config

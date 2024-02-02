from typing import Any

import gradio as gr


class UIConfigElement:
    """A base class for UI blocks that represent a part of a config."""

    def get_ui_components(self) -> list[gr.components.Component]:
        """Recursively return a list of all UI components that this UIConfigElement is composed of."""
        all_ui_components = []
        for attribute in vars(self).values():
            if isinstance(attribute, gr.components.Component):
                all_ui_components.append(attribute)
            elif isinstance(attribute, UIConfigElement):
                all_ui_components.extend(attribute.get_ui_components())
        return all_ui_components

    def update_ui_components_with_config_data(self, config) -> dict[gr.components.Component, Any]:
        """Produce a dictionary of UI components to their corresponding updated data from the config."""
        raise NotImplementedError()

    def update_config_with_ui_component_data(self, orig_config, ui_data: dict[gr.components.Component, Any]):
        """Update the orig_config with the data from the UI components. Return the updated config."""
        raise NotImplementedError()

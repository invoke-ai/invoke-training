import typing

import torch

from invoke_training.lora.layers import BaseLoRALayer


class LoRALayerCollection(torch.nn.Module):
    """A collection of LoRA layers (with names). Typically used to perform operations on a group of LoRA layers during
    training.
    """

    def __init__(self):
        super().__init__()

        # A torch.nn.ModuleDict may seem like a more natural choice here, but it does not allow keys that contain '.'
        # characters. Using a standard python dict is also inconvenient, because it would be ignored by torch.nn.Module
        # methods such as `.parameters()` and `.train()`.
        self._layers = torch.nn.ModuleList()
        self._names = []

    def add_layer(self, layer: BaseLoRALayer, name: str):
        self._layers.append(layer)
        self._names.append(name)

    def __len__(self):
        return len(self._layers)

    def get_lora_state_dict(self) -> typing.Dict[str, torch.Tensor]:
        """A custom alternative to .state_dict() that uses the layer names provided to add_layer(...) as key
        prefixes.
        """
        state_dict: typing.Dict[str, torch.Tensor] = {}

        for name, layer in zip(self._names, self._layers):
            layer_state_dict = layer.state_dict()
            for key, state in layer_state_dict.items():
                full_key = name + "." + key
                if full_key in state_dict:
                    raise RuntimeError(f"Multiple state elements map to the same key: '{full_key}'.")
                state_dict[full_key] = state

        return state_dict

import torch

from invoke_training.lora.layers import BaseLoRALayer


class LoRALayerCollection(torch.nn.Module):
    """A collection of LoRA layers (with names). Typically used to perform operations on a group of LoRA layers during
    training.
    """

    def __init__(self):
        super().__init__()

        self._layers = torch.nn.ModuleList()
        self._names = []

    def add_layer(self, layer: BaseLoRALayer, name: str):
        self._layers.append(layer)
        self._names.append(name)

    def __len__(self):
        return len(self._layers)

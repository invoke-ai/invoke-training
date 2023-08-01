import torch


class BaseLoRALayer(torch.nn.Module):
    """An interface that is implemented by all LoRA layers."""

    @classmethod
    def from_layer(
        cls,
        layer: torch.nn.Module,
        **kwargs,
    ):
        """Initialize a LoRA layer with dimensions that are compatible with 'layer'.
        Args:
            layer (torch.nn.Module): The existing layer whose in/out dimensions will be matched.
        Raises:
            TypeError: If layer has an unsupported type.
        Returns:
            cls: The new LoRA layer.
        """
        raise NotImplementedError("from_layer(...) is not yet implemented.")

import torch


class BaseLoRALayer(torch.nn.Module):
    """An interface that is implemented by all LoRA layers."""

    @classmethod
    def from_layer(
        cls,
        layer: torch.nn.Module,
        device: torch.device = None,
        dtype: torch.dtype = None,
        **kwargs,
    ):
        """Initialize a LoRA layer with dimensions that are compatible with 'layer'.
        Args:
            layer (torch.nn.Module): The existing layer whose in/out dimensions will be matched.
            device (torch.device, optional): The device to construct the new layer on.
            dtype (torch.dtype, optional): The dtype to construct the new layer with.
        Raises:
            TypeError: If layer has an unsupported type.
        Returns:
            cls: The new LoRA layer.
        """
        raise NotImplementedError("from_layer(...) is not yet implemented.")

import math

import torch

from invoke_training.lora.layers import BaseLoRALayer


class LoRAConvLayer(BaseLoRALayer):
    """An implementation of a conv LoRA layer based on the paper 'LoRA: Low-Rank Adaptation of Large Language Models'.
    (https://arxiv.org/pdf/2106.09685.pdf)
    """

    @property
    def conv_module(self):
        """The conv module to be set by child classes. One of torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d."""
        raise NotImplementedError(
            "LoRAConvLayer cannot be used directly. Use LoRAConv1dLayer, LoRAConv2dLayer, or LoRAConv3dLayer instead."
        )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int = 4,
        alpha: float = 1.0,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """Initialize a LoRAConvLayer.
        Args:
            in_channels (int): The number of channels expected on inputs to this layer.
            out_channels (int): The number of channels on outputs from this layer.
            rank (int, optional): The internal rank of the layer. See the paper for details.
            alpha (float, optional): A scaling factor that enables tuning the rank without having to adjust the learning
                rate. The recommendation from the paper is to set alpha equal to the first rank that you try and then do
                not tune it further. See the paper for more details.
            device (torch.device, optional): Device where weights will be initialized.
            dtype (torch.dtype, optional): Weight dtype.
        Raises:
            ValueError: If the rank is greater than either in_channels or out_channels.
        """
        super().__init__()

        if rank > min(in_channels, out_channels):
            raise ValueError(f"LoRA rank {rank} must be less than or equal to {min(in_channels, out_channels)}")

        self._down = self.conv_module(
            in_channels, rank, kernel_size=1, stride=1, bias=False, device=device, dtype=dtype
        )
        self._up = self.conv_module(rank, out_channels, kernel_size=1, stride=1, bias=False, device=device, dtype=dtype)

        # Register alpha as a buffer so that it is not trained, but still gets saved to the state_dict.
        self.register_buffer("alpha", torch.tensor(alpha, device=device, dtype=dtype))

        self._rank = rank

        self.reset_parameters()

    def reset_parameters(self):
        # This initialization is based on Microsoft's implementation:
        # https://github.com/microsoft/LoRA/blob/998cfe4d351f4d6b4a47f0921dec2397aa0b9dfe/loralib/layers.py#L279
        torch.nn.init.kaiming_uniform_(self._down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self._up.weight)

    @classmethod
    def from_layer(
        cls,
        layer: torch.nn.Module,
        rank: int = 4,
        alpha: float = 1.0,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """Initialize a LoRAConvLayer with dimensions that are compatible with `layer`.
        Args:
            layer (torch.nn.Module): The existing layer whose in/out dimensions will be matched.
            rank, alpha, device, dtype: These args are forwarded to __init__(...). If device or dtype are None, they
                will be inferred from `layer`.
        Raises:
            TypeError: If `layer` has an unsupported type.
        Returns:
            LoRAConvLayer: The new LoRAConvLayer.
        """
        if isinstance(layer, cls.conv_module):
            return cls(
                layer.in_channels,
                layer.out_channels,
                rank,
                alpha,
                layer.weight.device if device is None else device,
                layer.weight.dtype if dtype is None else dtype,
            )
        else:
            raise TypeError(f"'{__class__.__name__}' cannot be initialized from a layer of type '{type(layer)}'.")

    def forward(self, input: torch.Tensor):
        down_hidden = self._down(input)
        up_hidden = self._up(down_hidden)

        up_hidden *= self.alpha / self._rank

        return up_hidden


class LoRAConv1dLayer(LoRAConvLayer):
    conv_module = torch.nn.Conv1d


class LoRAConv2dLayer(LoRAConvLayer):
    conv_module = torch.nn.Conv2d


class LoRAConv3dLayer(LoRAConvLayer):
    conv_module = torch.nn.Conv3d

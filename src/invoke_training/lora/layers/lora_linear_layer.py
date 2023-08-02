import math

import torch

from invoke_training.lora.layers import BaseLoRALayer


class LoRALinearLayer(BaseLoRALayer):
    """An implementation of a linear LoRA layer based on the paper 'LoRA: Low-Rank Adaptation of Large Language Models'.
    (https://arxiv.org/pdf/2106.09685.pdf)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """Initialize a LoRALinearLayer.
        Args:
            in_features (int): Inputs to this layer will be expected to have shape (..., in_features).
            out_features (int): This layer will produce outputs with shape (..., out_features).
            rank (int, optional): The internal rank of the layer. See the paper for details.
            alpha (float, optional): A scaling factor that enables tuning the rank without having to adjust the learning
                rate. The recommendation from the paper is to set alpha equal to the first rank that you try and then do
                not tune it further. See the paper for more details.
            device (torch.device, optional): Device where weights will be initialized.
            dtype (torch.dtype, optional): Weight dtype.
        Raises:
            ValueError: If the rank is greater than either in_features or out_features.
        """
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less than or equal to {min(in_features, out_features)}")

        self._down = torch.nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self._up = torch.nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)

        # Register alpha as a buffer so that it is not trained, but still gets saved to the state_dict.
        self.register_buffer("alpha", torch.tensor(alpha, device=device, dtype=dtype))

        self._rank = rank

        self.reset_parameters()

    def reset_parameters(self):
        # This initialization is based on Microsoft's implementation:
        # https://github.com/microsoft/LoRA/blob/998cfe4d351f4d6b4a47f0921dec2397aa0b9dfe/loralib/layers.py#L123
        torch.nn.init.kaiming_uniform_(self._down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self._up.weight)

    @classmethod
    def from_layer(
        cls,
        layer: torch.nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """Initialize a LoRALinearLayer with dimensions that are compatible with 'layer'.
        Args:
            layer (torch.nn.Linear): The existing layer whose in/out dimensions will be matched.
            rank, alpha, device, dtype: These args are forwarded to __init__(...). If device or dtype are None, they
                will be inferred from 'layer'.
        Raises:
            TypeError: If 'layer' has an unsupported type.
        Returns:
            LoRALinearLayer: The new LoRALinearLayer.
        """
        if isinstance(layer, torch.nn.Linear):
            return cls(
                layer.in_features,
                layer.out_features,
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

import torch


class LoRABlock(torch.nn.Module):
    """A wrapper block that combines the outputs of an 'original' module and a parallel 'LoRA' layer."""

    def __init__(self, original_module: torch.nn.Module, lora_layer: torch.nn.Module, lora_multiplier: float = 1.0):
        """Initialize a LoRABlock.
        Args:
            original_module (torch.nn.Module): The original module.
            lora_layer (torch.nn.Module): The LoRA layer.
            lora_multiplier (float, optional): A multiplier applied to the LoRA layer output before adding it to the
                original module output. Defaults to 1.0.
        """
        super().__init__()

        self.original_module = original_module
        self.lora_layer = lora_layer
        self.lora_multiplier = lora_multiplier

    def forward(self, input, *args, **kwargs):
        return self.original_module(input, *args, **kwargs) + self.lora_multiplier * self.lora_layer(input)

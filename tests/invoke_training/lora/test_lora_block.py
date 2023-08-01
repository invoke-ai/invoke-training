import torch

from invoke_training.lora.lora_block import LoRABlock


def test_lora_module_multiplier():
    """A basic test that the lora_multitplier param is being applied correctly."""
    original = torch.nn.Linear(1, 1, bias=False)
    original.weight = torch.nn.Parameter(torch.Tensor([[1]]))

    lora = torch.nn.Linear(1, 1, bias=False)
    lora.weight = torch.nn.Parameter(torch.Tensor([[2]]))

    layer = LoRABlock(original, lora, lora_multiplier=2)

    with torch.no_grad():
        y = layer.forward(torch.Tensor([[1]]))

    # expected: y = (1 * in) + 2 * (2 * in) = 5
    torch.testing.assert_close(y, torch.Tensor([[5]]))

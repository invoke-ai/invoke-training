import pytest
import torch

from invoke_training.lora.layers import LoRALinearLayer


def test_lora_linear_layer_output_dim():
    """Test LoRALinearLayer produces an output with the expected dimension."""
    batch_size = 10
    in_features = 8
    out_features = 16
    layer = LoRALinearLayer(in_features, out_features, 2)

    x = torch.rand((batch_size, in_features))
    with torch.no_grad():
        y = layer.forward(x)

    assert y.shape == (batch_size, out_features)


def test_lora_linear_layer_invalid_input_dim():
    """Test that LoRALinearLayer throws an exception if it receives an input with invalid dimensions."""
    in_features = 8
    out_features = 16
    layer = LoRALinearLayer(in_features, out_features, 2)

    x = torch.rand((10, in_features + 1))  # Bad input dimension.

    with pytest.raises(RuntimeError):
        _ = layer.forward(x)


def test_lora_linear_layer_zero_after_init():
    """Test that a newly-initialized LoRALinearLayer produces all zeros before it is trained."""
    batch_size = 10
    in_features = 8
    out_features = 16
    layer = LoRALinearLayer(in_features, out_features, 2)

    x = torch.rand((batch_size, in_features))
    with torch.no_grad():
        y = layer.forward(x)

    assert not torch.allclose(x, torch.Tensor([0.0]), rtol=0.0)  # The random input was non-zero.
    assert torch.allclose(y, torch.Tensor([0.0]), rtol=0.0)  # The untrained outputs are zero.


def test_lora_linear_layer_from_layer():
    """Test that a LoRALinearLayer can be initialized correctly from a torch.nn.Linear layer."""
    batch_size = 10
    in_features = 4
    out_features = 16
    original_layer = torch.nn.Linear(in_features, out_features)

    lora_layer: LoRALinearLayer = LoRALinearLayer.from_layer(original_layer)

    x = torch.rand((batch_size, in_features))
    with torch.no_grad():
        y = lora_layer.forward(x)

    assert y.shape == (batch_size, out_features)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_lora_linear_layer_from_layer_inherit_device_and_dtype(dtype):
    """Test that when a LoRALinearLayer is initialized with from_layer(...), it correctly inherits the device and
    dtype.
    """
    batch_size = 10
    in_features = 4
    out_features = 16
    original_layer = torch.nn.Linear(in_features, out_features, device=torch.device("cuda"), dtype=dtype)

    lora_layer: LoRALinearLayer = LoRALinearLayer.from_layer(original_layer)

    x = torch.rand((batch_size, in_features), device=torch.device("cuda"), dtype=dtype)
    with torch.no_grad():
        y = lora_layer.forward(x)

    assert y.shape == (batch_size, out_features)
    # Assert that lora_layer's internal layers have correct device and dtype.
    assert lora_layer._down.weight.device == original_layer.weight.device
    assert lora_layer._down.weight.dtype == original_layer.weight.dtype
    assert lora_layer._up.weight.device == original_layer.weight.device
    assert lora_layer._up.weight.dtype == original_layer.weight.dtype


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_lora_linear_layer_from_layer_override_device_and_dtype(dtype):
    """Test that when a LoRALinearLayer is initialized with from_layer(...), the device and dtype can be overriden."""
    # Original layer has dtype float32 on CPU.
    original_layer = torch.nn.Linear(4, 16, dtype=torch.float32)

    target_device = torch.device("cuda:0")
    lora_layer: LoRALinearLayer = LoRALinearLayer.from_layer(original_layer, device=target_device, dtype=dtype)

    # Assert that lora_layer's internal layers have correct device and dtype.
    assert lora_layer._down.weight.device == target_device
    assert lora_layer._down.weight.dtype == dtype
    assert lora_layer._up.weight.device == target_device
    assert lora_layer._up.weight.dtype == dtype

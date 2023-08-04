import typing

import pytest
import torch

from invoke_training.lora.layers import (
    LoRAConv1dLayer,
    LoRAConv2dLayer,
    LoRAConv3dLayer,
)
from invoke_training.lora.layers.lora_conv_layer import LoRAConvLayer


def test_lora_conv_layer_initialize_base_class():
    """Test that attempting to directly initialize a LoRAConvLayer raise a NotImplementedError."""
    with pytest.raises(NotImplementedError):
        _ = LoRAConvLayer(4, 8)


@pytest.mark.parametrize(
    ["lora_conv_cls", "conv_dims"], [(LoRAConv1dLayer, 1), (LoRAConv2dLayer, 2), (LoRAConv3dLayer, 3)]
)
class TestLoRAConvLayers:
    """Test class for applying tests to each of the LoRAConv*Layer classes."""

    def test_lora_conv_layer_output_dim(self, lora_conv_cls: typing.Type[LoRAConvLayer], conv_dims: int):
        """Test that LoRAConv*Layer produces an output with the expected dimensions."""
        batch_size = 10
        in_channels = 8
        out_channels = 16
        layer = lora_conv_cls(in_channels, out_channels)

        in_shape = (batch_size, in_channels) + (5,) * conv_dims
        x = torch.rand(in_shape)
        with torch.no_grad():
            y = layer(x)

        expected_out_shape = (batch_size, out_channels) + (5,) * conv_dims
        assert y.shape == expected_out_shape

    def test_lora_conv_layer_invalid_input_dim(self, lora_conv_cls: typing.Type[LoRAConvLayer], conv_dims: int):
        """Test that LoRAConv*Layer raises an exception if it receives an input with invalid dimensions."""
        batch_size = 10
        in_channels = 8
        out_channels = 16
        layer = lora_conv_cls(in_channels, out_channels)

        in_shape = (batch_size, in_channels + 1) + (5,) * conv_dims  # Bad input dimension.
        x = torch.rand(in_shape)
        with pytest.raises(RuntimeError):
            _ = layer(x)

    def test_lora_conv_layer_zero_after_init(self, lora_conv_cls: typing.Type[LoRAConvLayer], conv_dims: int):
        """Test that a newly-initialized LoRAConv*Layer produces all zeros before it is trained."""
        batch_size = 10
        in_channels = 8
        out_channels = 16
        layer = lora_conv_cls(in_channels, out_channels)

        in_shape = (batch_size, in_channels) + (5,) * conv_dims
        x = torch.rand(in_shape)
        with torch.no_grad():
            y = layer(x)

        assert not torch.allclose(x, torch.Tensor([0.0]), rtol=0.0)  # The random input was non-zero.
        assert torch.allclose(y, torch.Tensor([0.0]), rtol=0.0)  # The untrained outputs are zero.

    def test_lora_conv_layer_from_layer(self, lora_conv_cls: typing.Type[LoRAConvLayer], conv_dims: int):
        """Test that a LoRAConv*Layer can be initialized correctly from a torch.nn.Conv* layer."""
        batch_size = 10
        in_channels = 8
        out_channels = 16
        original_layer = lora_conv_cls.conv_module(in_channels, out_channels, kernel_size=3)

        lora_layer = lora_conv_cls.from_layer(original_layer)

        in_shape = (batch_size, in_channels) + (5,) * conv_dims
        x = torch.rand(in_shape)
        with torch.no_grad():
            y = lora_layer(x)

        expected_out_shape = (batch_size, out_channels) + (5,) * conv_dims
        assert y.shape == expected_out_shape

    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_lora_conv_layer_from_layer_inherit_device_and_dtype(
        self, lora_conv_cls: typing.Type[LoRAConvLayer], conv_dims: int, dtype: torch.dtype
    ):
        """Test that when a LoRAConv*Layer is initialized with from_layer(...), it correctly inherits the device and
        dtype.
        """
        batch_size = 10
        in_channels = 8
        out_channels = 16
        original_layer = lora_conv_cls.conv_module(
            in_channels, out_channels, kernel_size=3, device=torch.device("cuda"), dtype=dtype
        )

        lora_layer = lora_conv_cls.from_layer(original_layer)

        in_shape = (batch_size, in_channels) + (5,) * conv_dims
        x = torch.rand(in_shape, device=torch.device("cuda"), dtype=dtype)
        with torch.no_grad():
            y = lora_layer(x)

        expected_out_shape = (batch_size, out_channels) + (5,) * conv_dims
        assert y.shape == expected_out_shape
        # Assert that lora_layer's internal layers have correct device and dtype.
        assert lora_layer._down.weight.device == original_layer.weight.device
        assert lora_layer._down.weight.dtype == original_layer.weight.dtype
        assert lora_layer._up.weight.device == original_layer.weight.device
        assert lora_layer._up.weight.dtype == original_layer.weight.dtype

    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_lora_conv_layer_from_layer_override_device_and_dtype(
        self, lora_conv_cls: typing.Type[LoRAConvLayer], conv_dims: int, dtype: torch.dtype
    ):
        """Test that when a LoRAConv*Layer is initialized with from_layer(...), the device and dtype can be
        overriden."""
        batch_size = 10
        in_channels = 8
        out_channels = 16
        # Original layer has dtype float32 on CPU.
        original_layer = lora_conv_cls.conv_module(in_channels, out_channels, kernel_size=3, dtype=torch.float32)

        target_device = torch.device("cuda:0")
        lora_layer = lora_conv_cls.from_layer(original_layer, device=target_device, dtype=dtype)

        in_shape = (batch_size, in_channels) + (5,) * conv_dims
        x = torch.rand(in_shape, device=torch.device("cuda"), dtype=dtype)
        with torch.no_grad():
            y = lora_layer(x)

        expected_out_shape = (batch_size, out_channels) + (5,) * conv_dims
        assert y.shape == expected_out_shape
        # Assert that lora_layer's internal layers have correct device and dtype.
        assert lora_layer._down.weight.device == target_device
        assert lora_layer._down.weight.dtype == dtype
        assert lora_layer._up.weight.device == target_device
        assert lora_layer._up.weight.dtype == dtype

    def test_lora_conv_layer_state_dict_roundtrip(self, lora_conv_cls: typing.Type[LoRAConvLayer], conv_dims: int):
        original_layer = lora_conv_cls(8, 16)

        state_dict = original_layer.state_dict()

        roundtrip_layer = lora_conv_cls(8, 16, alpha=2.0)

        # Prior to loading the state_dict, the roundtrip_layer is different than the original_layer.
        # (We don't compare the _up layer, because it is initialized to zeros so should match already.)
        assert not torch.allclose(roundtrip_layer._down.weight, original_layer._down.weight)
        assert not torch.allclose(roundtrip_layer.alpha, original_layer.alpha)

        roundtrip_layer.load_state_dict(state_dict)

        # After loading the state_dict the roundtrip_layer and original_layer match.
        assert torch.allclose(roundtrip_layer._down.weight, original_layer._down.weight)
        assert torch.allclose(roundtrip_layer._up.weight, original_layer._up.weight)
        assert torch.allclose(roundtrip_layer.alpha, original_layer.alpha)

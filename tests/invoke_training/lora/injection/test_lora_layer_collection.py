import pytest

from invoke_training.lora.injection.lora_layer_collection import LoRALayerCollection
from invoke_training.lora.layers import LoRALinearLayer


def test_lora_layer_collection_state_dict():
    """Test the behavior of LoRALayerCollection.get_lora_state_dict()."""
    lora_layers = LoRALayerCollection()

    lora_layers.add_layer(LoRALinearLayer(8, 16), "lora_layer_1")
    lora_layers.add_layer(LoRALinearLayer(16, 32), "lora_layer_2")

    state_dict = lora_layers.get_lora_state_dict()

    expected_state_keys = {
        "lora_layer_1._down.weight",
        "lora_layer_1._up.weight",
        "lora_layer_2._down.weight",
        "lora_layer_2._up.weight",
    }
    assert set(state_dict.keys()) == expected_state_keys


def test_lora_layer_collection_state_dict_conflicting_keys():
    """Test that LoRALayerCollection.get_lora_state_dict() raises an exception if state Tensors have conflicting
    keys.
    """
    lora_layers = LoRALayerCollection()

    lora_layers.add_layer(LoRALinearLayer(8, 16), "lora_layer_1")
    lora_layers.add_layer(LoRALinearLayer(16, 32), "lora_layer_1")  # Insert same layer type with same key.

    with pytest.raises(RuntimeError):
        _ = lora_layers.get_lora_state_dict()

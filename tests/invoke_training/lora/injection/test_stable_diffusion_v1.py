import pytest
import torch
from diffusers.models import UNet2DConditionModel

from invoke_training.lora.injection.stable_diffusion_v1 import (
    convert_lora_state_dict_to_kohya_format_sd1,
    inject_lora_into_unet_sd1,
)


@pytest.mark.loads_model
def test_inject_lora_into_unet_sd1_smoke():
    """Smoke test of inject_lora_into_unet_sd1(...) on full SD 1.5 model."""
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet", local_files_only=True
    )

    lora_layers = inject_lora_into_unet_sd1(unet)

    # These assertions are based on a manual check of the injected layers and comparison against the behaviour of
    # kohya_ss. They are included here to force another manual review after any future breaking change.
    assert len(lora_layers) == 160
    # assert len(lora_layers) == 192 # TODO(ryand): Enable this check once conv layers are added.
    for layer_name in lora_layers._names:
        assert layer_name.endswith(("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"))


@pytest.mark.loads_model
def test_convert_lora_state_dict_to_kohya_format_sd1_smoke():
    """Smoke test of convert_lora_state_dict_to_kohya_format_sd1(...) with full SD 1.5 model."""
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet", local_files_only=True
    )

    lora_layers = inject_lora_into_unet_sd1(unet)
    lora_state_dict = lora_layers.get_lora_state_dict()
    kohya_state_dict = convert_lora_state_dict_to_kohya_format_sd1(lora_state_dict)

    # These assertions are based on a manual check of the injected layers and comparison against the behaviour of
    # kohya_ss. They are included here to force another manual review after any future breaking change.
    assert len(kohya_state_dict) == 160 * 3
    for key in kohya_state_dict.keys():
        assert key.startswith("lora_unet_")
        assert key.endswith((".lora_down.weight", ".lora_up.weight", ".alpha"))


def test_convert_lora_state_dict_to_kohya_format_sd1():
    """Basic test of convert_lora_state_dict_to_kohya_format_sd1(...)."""
    down_weight = torch.Tensor(4, 2)
    up_weight = torch.Tensor(2, 4)
    alpha = torch.Tensor([1.0])
    in_state_dict = {
        "lora_unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q._down.weight": down_weight,
        "lora_unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q._up.weight": up_weight,
        "lora_unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.alpha": alpha,
    }

    out_state_dict = convert_lora_state_dict_to_kohya_format_sd1(in_state_dict)

    expected_out_state_dict = {
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight": down_weight,
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight": up_weight,
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.alpha": alpha,
    }

    assert out_state_dict == expected_out_state_dict


def test_convert_lora_state_dict_to_kohya_format_sd1_unexpected_key():
    """Test that convert_lora_state_dict_to_kohya_format_sd1(...) raises an exception if it receives an unexpected
    key.
    """
    in_state_dict = {
        "lora_unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q._down.unexpected": torch.Tensor(4, 2),
    }

    with pytest.raises(ValueError):
        _ = convert_lora_state_dict_to_kohya_format_sd1(in_state_dict)


def test_convert_lora_state_dict_to_kohya_format_sd1_conflicting_keys():
    """Test that convert_lora_state_dict_to_kohya_format_sd1(...) raises an exception if multiple keys map to the same
    output key.
    """
    # Note: There are differences in the '.' and '_' characters of these keys, but they both map to the same output
    # kohya_ss keys.
    in_state_dict = {
        "lora_unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q._down.weight": torch.Tensor(4, 2),
        "lora_unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1_to_q._down.weight": torch.Tensor(4, 2),
    }

    with pytest.raises(RuntimeError):
        _ = convert_lora_state_dict_to_kohya_format_sd1(in_state_dict)

import pytest
import torch
from diffusers.models import UNet2DConditionModel
from transformers import CLIPTextModel

from invoke_training.lora.injection.stable_diffusion import (
    convert_lora_state_dict_to_kohya_format,
    inject_lora_into_clip_text_encoder,
    inject_lora_into_unet,
)


@pytest.mark.loads_model
@pytest.mark.parametrize(
    ["model_name", "revision", "expected_num_layers"],
    [
        ("runwayml/stable-diffusion-v1-5", "c9ab35ff5f2c362e9e22fbafe278077e196057f0", 192),
        ("stabilityai/stable-diffusion-xl-base-1.0", "47cd5302d866fa60cf8fb81f0e34d42e38f6100c", 722),
    ],
)
def test_inject_lora_into_unet_smoke(model_name: str, revision: str, expected_num_layers: int):
    """Smoke test of inject_lora_into_unet(...)."""
    unet = UNet2DConditionModel.from_pretrained(
        model_name,
        subfolder="unet",
        local_files_only=True,
        revision=revision,
    )
    lora_layers = inject_lora_into_unet(unet)

    # These assertions are based on a manual check of the injected layers and comparison against the behaviour of
    # kohya_ss. They are included here to force another manual review after any future breaking change.
    assert len(lora_layers) == expected_num_layers
    for layer_name in lora_layers._names:
        assert layer_name.endswith(
            ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", ".proj_in", ".proj_out")
        )


@pytest.mark.loads_model
@pytest.mark.parametrize(
    ["model_name", "revision", "expected_num_layers"],
    [
        ("runwayml/stable-diffusion-v1-5", "c9ab35ff5f2c362e9e22fbafe278077e196057f0", 278),
        ("stabilityai/stable-diffusion-xl-base-1.0", "47cd5302d866fa60cf8fb81f0e34d42e38f6100c", 788),
    ],
)
def test_inject_lora_into_unet_non_attention_layers_smoke(model_name: str, revision: str, expected_num_layers: int):
    """Smoke test of inject_lora_into_unet(..., include_non_attention_blocks=True)."""
    unet = UNet2DConditionModel.from_pretrained(
        model_name,
        subfolder="unet",
        local_files_only=True,
        revision=revision,
    )
    lora_layers = inject_lora_into_unet(unet, include_non_attention_blocks=True)

    # These assertions are based on a manual check of the injected layers and comparison against the behaviour of
    # kohya_ss. They are included here to force another manual review after any future breaking change.
    assert len(lora_layers) == expected_num_layers
    for layer_name in lora_layers._names:
        assert layer_name.endswith(
            (
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "ff.net.0.proj",
                "ff.net.2",
                ".proj_in",
                ".proj_out",
                ".conv1",
                ".conv2",
                ".time_emb_proj",
                ".conv",
                ".conv_shortcut",
            )
        )


@pytest.mark.loads_model
@pytest.mark.parametrize(
    ["model_name", "revision", "text_encoder_name", "expected_num_layers"],
    [
        ("stabilityai/stable-diffusion-xl-base-1.0", "47cd5302d866fa60cf8fb81f0e34d42e38f6100c", "text_encoder", 72),
        ("stabilityai/stable-diffusion-xl-base-1.0", "47cd5302d866fa60cf8fb81f0e34d42e38f6100c", "text_encoder_2", 192),
        ("runwayml/stable-diffusion-v1-5", "c9ab35ff5f2c362e9e22fbafe278077e196057f0", "text_encoder", 72),
    ],
)
def test_inject_lora_into_clip_text_encoder_smoke(model_name, revision, text_encoder_name, expected_num_layers):
    """Smoke test of inject_lora_into_clip_text_encoder(...)."""
    text_encoder = CLIPTextModel.from_pretrained(
        model_name,
        subfolder=text_encoder_name,
        local_files_only=True,
        revision=revision,
    )

    lora_layers = inject_lora_into_clip_text_encoder(text_encoder)

    # These assertions are based on a manual check of the injected layers and comparison against the behaviour of
    # kohya_ss. They are included here to force another manual review after any future breaking change.
    assert len(lora_layers) == expected_num_layers
    for layer_name in lora_layers._names:
        assert layer_name.endswith(("mlp.fc1", "mlp.fc2", "k_proj", "out_proj", "q_proj", "v_proj"))


@pytest.mark.loads_model
@pytest.mark.loads_model
@pytest.mark.parametrize(
    ["model_name", "revision", "expected_num_layers"],
    [
        ("runwayml/stable-diffusion-v1-5", "c9ab35ff5f2c362e9e22fbafe278077e196057f0", 192),
        ("stabilityai/stable-diffusion-xl-base-1.0", "47cd5302d866fa60cf8fb81f0e34d42e38f6100c", 722),
    ],
)
def test_convert_lora_state_dict_to_kohya_format_smoke(model_name: str, revision: str, expected_num_layers: int):
    """Smoke test of convert_lora_state_dict_to_kohya_format(...) with full SD 1.5 model."""
    unet = UNet2DConditionModel.from_pretrained(
        model_name,
        subfolder="unet",
        local_files_only=True,
        revision=revision,
    )
    lora_layers = inject_lora_into_unet(unet)
    lora_state_dict = lora_layers.get_lora_state_dict()
    kohya_state_dict = convert_lora_state_dict_to_kohya_format(lora_state_dict)

    # These assertions are based on a manual check of the injected layers and comparison against the behaviour of
    # kohya_ss. They are included here to force another manual review after any future breaking change.
    assert len(kohya_state_dict) == expected_num_layers * 3
    for key in kohya_state_dict.keys():
        assert key.startswith("lora_unet_")
        assert key.endswith((".lora_down.weight", ".lora_up.weight", ".alpha"))


def test_convert_lora_state_dict_to_kohya_format():
    """Basic test of convert_lora_state_dict_to_kohya_format(...)."""
    down_weight = torch.Tensor(4, 2)
    up_weight = torch.Tensor(2, 4)
    alpha = torch.Tensor([1.0])
    in_state_dict = {
        "lora_unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q._down.weight": down_weight,
        "lora_unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q._up.weight": up_weight,
        "lora_unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.alpha": alpha,
    }

    out_state_dict = convert_lora_state_dict_to_kohya_format(in_state_dict)

    expected_out_state_dict = {
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight": down_weight,
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight": up_weight,
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.alpha": alpha,
    }

    assert out_state_dict == expected_out_state_dict


def test_convert_lora_state_dict_to_kohya_format_unexpected_key():
    """Test that convert_lora_state_dict_to_kohya_format(...) raises an exception if it receives an unexpected
    key.
    """
    in_state_dict = {
        "lora_unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q._down.unexpected": torch.Tensor(4, 2),
    }

    with pytest.raises(ValueError):
        _ = convert_lora_state_dict_to_kohya_format(in_state_dict)


def test_convert_lora_state_dict_to_kohya_format_conflicting_keys():
    """Test that convert_lora_state_dict_to_kohya_format(...) raises an exception if multiple keys map to the same
    output key.
    """
    # Note: There are differences in the '.' and '_' characters of these keys, but they both map to the same output
    # kohya_ss keys.
    in_state_dict = {
        "lora_unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q._down.weight": torch.Tensor(4, 2),
        "lora_unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1_to_q._down.weight": torch.Tensor(4, 2),
    }

    with pytest.raises(RuntimeError):
        _ = convert_lora_state_dict_to_kohya_format(in_state_dict)

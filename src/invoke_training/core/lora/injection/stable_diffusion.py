import typing

import torch
from diffusers.models import Transformer2DModel, UNet2DConditionModel
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.resnet import Downsample2D, ResnetBlock2D, Upsample2D
from transformers import CLIPTextModel
from transformers.models.clip.modeling_clip import CLIPMLP, CLIPAttention

from invoke_training.core.lora.injection.lora_layer_collection import LoRALayerCollection
from invoke_training.core.lora.injection.utils import inject_lora_layers
from invoke_training.core.lora.layers import LoRAConv2dLayer, LoRALinearLayer


def inject_lora_into_unet(
    unet: UNet2DConditionModel, include_non_attention_blocks: bool = False, lora_rank_dim: int = 4
) -> LoRALayerCollection:
    """Inject LoRA layers into a Stable Diffusion UNet model.

    Args:
        unet (UNet2DConditionModel): The UNet model to inject LoRA layers into.
        include_non_attention_blocks (bool, optional): Whether to inject LoRA layers into the linear/conv layers of the
            non-attention blocks (`ResnetBlock2D`, `Downsample2D`, `Upsample2D`). Defaults to False.
        lora_rank_dim (int, optional): The LoRA layer rank dimension.
    Returns:
        LoRALayerCollection: The LoRA layers that were added to the UNet.
    """
    include_descendants_of = {Transformer2DModel}
    if include_non_attention_blocks:
        include_descendants_of.update({ResnetBlock2D, Downsample2D, Upsample2D})

    lora_layers = inject_lora_layers(
        module=unet,
        lora_map={
            torch.nn.Linear: LoRALinearLayer,
            LoRACompatibleLinear: LoRALinearLayer,
            torch.nn.Conv2d: LoRAConv2dLayer,
            LoRACompatibleConv: LoRAConv2dLayer,
        },
        include_descendants_of=include_descendants_of,
        exclude_descendants_of=None,
        prefix="lora_unet",
        dtype=torch.float32,
        lora_rank_dim=lora_rank_dim,
    )

    return lora_layers


def inject_lora_into_clip_text_encoder(text_encoder: CLIPTextModel, prefix: str = "lora_te", lora_rank_dim: int = 4):
    lora_layers = inject_lora_layers(
        module=text_encoder,
        lora_map={
            torch.nn.Linear: LoRALinearLayer,
            torch.nn.Conv2d: LoRAConv2dLayer,
        },
        include_descendants_of={CLIPAttention, CLIPMLP},
        exclude_descendants_of=None,
        prefix=prefix,
        dtype=torch.float32,
        lora_rank_dim=lora_rank_dim,
    )

    return lora_layers


def convert_invoke_to_kohya_lora_state_dict(
    invoke_state_dict: typing.Dict[str, torch.Tensor],
) -> typing.Dict[str, torch.Tensor]:
    """Convert a Stable Diffusion LoRA state_dict from internal invoke-training format to kohya_ss format.

    Args:
        invoke_state_dict (typing.Dict[str, torch.Tensor]): LoRA layer state_dict in invoke-training format.

    Raises:
        ValueError: If invoke_state_dict contains unexpected keys.
        RuntimeError: If two input keys map to the same output kohya_ss key.

    Returns:
        typing.Dict[str, torch.Tensor]: LoRA layer state_dict in kohya_ss format.
    """
    invoke_to_kohya = build_invoke_to_kohya_key_map(invoke_state_dict)

    kohya_state_dict = {}
    for key, val in invoke_state_dict.items():
        kohya_key = invoke_to_kohya[key]
        kohya_state_dict[kohya_key] = val

    return kohya_state_dict


def convert_kohya_to_invoke_lora_state_dict(
    invoke_state_dict: typing.Dict[str, torch.Tensor], kohya_state_dict: typing.Dict[str, torch.Tensor]
) -> typing.Dict[str, torch.Tensor]:
    invoke_to_kohya = build_invoke_to_kohya_key_map(invoke_state_dict)

    new_invoke_state_dict = {}
    for key in invoke_state_dict:
        new_invoke_state_dict[key] = kohya_state_dict[invoke_to_kohya[key]]

    return new_invoke_state_dict


def build_invoke_to_kohya_key_map(invoke_state_dict: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, str]:
    all_kohya_keys = set()
    invoke_to_kohya = {}

    # The following logic converts state_dict keys from the internal invoke-training format to kohya_ss format.
    # Example conversion:
    # from: 'lora_unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q._down.weight'
    # to:   'lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight'
    for key in invoke_state_dict:
        if key.endswith("._up.weight"):
            key_start = key.removesuffix("._up.weight")
            key_end = ".lora_up.weight"
        elif key.endswith("._down.weight"):
            key_start = key.removesuffix("._down.weight")
            key_end = ".lora_down.weight"
        elif key.endswith(".alpha"):
            key_start = key.removesuffix(".alpha")
            key_end = ".alpha"
        else:
            raise ValueError(f"Unexpected key in state_dict: '{key}'.")

        kohya_key = key_start.replace(".", "_") + key_end

        if kohya_key in all_kohya_keys:
            raise RuntimeError(f"Multiple input keys map to the same kohya_ss key: '{kohya_key}'.")
        all_kohya_keys.add(kohya_key)

        invoke_to_kohya[key] = kohya_key

    return invoke_to_kohya

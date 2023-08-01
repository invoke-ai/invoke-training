import torch
from diffusers.models import Transformer2DModel, UNet2DConditionModel
from diffusers.models.attention import FeedForward
from diffusers.models.lora import LoRACompatibleLinear

from invoke_training.lora.injection.lora_layer_collection import LoRALayerCollection
from invoke_training.lora.injection.utils import inject_lora_layers
from invoke_training.lora.layers import LoRALinearLayer


def inject_lora_into_unet_sd1(unet: UNet2DConditionModel) -> LoRALayerCollection:
    """Inject LoRA layers into a Stable Diffusion v1 UNet model.

    Args:
        unet (UNet2DConditionModel): The UNet model to inject LoRA layers into.

    Returns:
        LoRALayerCollection: The LoRA layers that were added to the UNet.
    """

    # This combination of including Transformer2DModel descendants and excluding FeedForward descendants is designed to
    # mimic how kohya_ss handles Linear LoRA layers. After initial testing, this will be expanded to include the
    # FeedForward layers as well.
    lora_layers = inject_lora_layers(
        module=unet,
        lora_map={torch.nn.Linear: LoRALinearLayer, LoRACompatibleLinear: LoRALinearLayer},
        include_descendants_of={Transformer2DModel},
        exclude_descendants_of={FeedForward},
    )

    return lora_layers


def convert_lora_state_dict_to_kohya_format_sd1(state_dict):
    pass

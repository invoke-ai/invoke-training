import pytest
from diffusers.models import UNet2DConditionModel

from invoke_training.lora.injection.stable_diffusion_v1 import inject_lora_into_unet_sd1


@pytest.mark.loads_model
def test_inject_lora_into_unet_sd1():
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet", local_files_only=True
    )

    lora_layers = inject_lora_into_unet_sd1(unet)

    # These assertions are based on a manual check of the injected layers and comparison against the behaviour of
    # kohya_ss. They are included here to force another manual review after any future breaking change.
    assert len(lora_layers) == 128
    for layer_name in lora_layers._names:
        assert layer_name.endswith(("to_q", "to_k", "to_v", "to_out.0"))

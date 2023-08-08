import pytest
from transformers import PretrainedConfig

from invoke_training.training.shared.base_model_version import (
    BaseModelVersionEnum,
    get_base_model_version,
)


@pytest.mark.loads_model
@pytest.mark.parametrize(
    ["diffusers_model_name", "expected_version"],
    [
        ("runwayml/stable-diffusion-v1-5", BaseModelVersionEnum.STABLE_DIFFUSION_V1),
        ("stabilityai/stable-diffusion-2-1", BaseModelVersionEnum.STABLE_DIFFUSION_V2),
        ("stabilityai/stable-diffusion-xl-base-1.0", BaseModelVersionEnum.STABLE_DIFFUSION_SDXL_BASE),
        ("stabilityai/stable-diffusion-xl-refiner-1.0", BaseModelVersionEnum.STABLE_DIFFUSION_SDXL_REFINER),
    ],
)
def test_get_base_model_version(diffusers_model_name: str, expected_version: BaseModelVersionEnum):
    # Check if the diffusers_model_name model is downloaded and xfail if not.
    # This check ensures that users don't have to download all of the test models just to run the test suite.
    try:
        _ = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path=diffusers_model_name,
            subfolder="unet",
            local_files_only=True,
        )
    except OSError:
        pytest.xfail(f"'{diffusers_model_name}' is not downloaded.")

    version = get_base_model_version(diffusers_model_name)
    assert version == expected_version

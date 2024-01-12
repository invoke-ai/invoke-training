import pytest
from transformers import PretrainedConfig

from invoke_training.training._shared.stable_diffusion.base_model_version import (
    BaseModelVersionEnum,
    check_base_model_version,
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
    """Test get_base_model_version(...) with one test model for each supported version."""
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


@pytest.mark.loads_model
def test_check_base_model_version_pass():
    """Test that check_base_model_version(...) does not raise an Exception when the model is valid."""
    check_base_model_version({BaseModelVersionEnum.STABLE_DIFFUSION_V1}, "runwayml/stable-diffusion-v1-5")


@pytest.mark.loads_model
def test_check_base_model_version_fail():
    """Test that check_base_model_version(...) raises a ValueError when the model is invalid."""
    with pytest.raises(ValueError):
        check_base_model_version({BaseModelVersionEnum.STABLE_DIFFUSION_V2}, "runwayml/stable-diffusion-v1-5")

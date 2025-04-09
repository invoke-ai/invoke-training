from enum import Enum

from transformers import PretrainedConfig


class BaseModelVersionEnum(Enum):
    STABLE_DIFFUSION_V1 = 1
    STABLE_DIFFUSION_V2 = 2
    STABLE_DIFFUSION_SDXL_BASE = 3
    STABLE_DIFFUSION_SDXL_REFINER = 4


def get_base_model_version(
    diffusers_model_name: str, revision: str = "main", local_files_only: bool = True
) -> BaseModelVersionEnum:
    """Returns the `BaseModelVersionEnum` of a diffusers model.

    Args:
        diffusers_model_name (str): The diffusers model name (on Hugging Face Hub).
        revision (str, optional): The model revision (branch or commit hash). Defaults to "main".

    Raises:
        Exception: If the base model version can not be determined.

    Returns:
        BaseModelVersionEnum: The detected base model version.
    """
    unet_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path=diffusers_model_name,
        revision=revision,
        subfolder="unet",
        local_files_only=local_files_only,
    )

    # This logic was copied from
    # https://github.com/invoke-ai/InvokeAI/blob/e77400ab62d24acbdf2f48a7427705e7b8b97e4a/invokeai/backend/model_management/model_probe.py#L412-L421
    # This seems fragile. If you see this and know of a better way to detect the base model version, your contribution
    # would be welcome.
    if unet_config.cross_attention_dim == 768:
        return BaseModelVersionEnum.STABLE_DIFFUSION_V1
    elif unet_config.cross_attention_dim == 1024:
        return BaseModelVersionEnum.STABLE_DIFFUSION_V2
    elif unet_config.cross_attention_dim == 1280:
        return BaseModelVersionEnum.STABLE_DIFFUSION_SDXL_REFINER
    elif unet_config.cross_attention_dim == 2048:
        return BaseModelVersionEnum.STABLE_DIFFUSION_SDXL_BASE
    else:
        raise Exception(
            "Failed to determine base model version. UNet cross_attention_dim has unexpected value: "
            f"'{unet_config.cross_attention_dim}'."
        )


def check_base_model_version(
    allowed_versions: set[BaseModelVersionEnum],
    diffusers_model_name: str,
    revision: str = "main",
    local_files_only: bool = True,
):
    """Helper function that checks if a diffusers model is compatible with a set of base model versions.

    Args:
        allowed_versions (set[BaseModelVersionEnum]): The set of allowed base model versions.
        diffusers_model_name (str): The diffusers model name (on Hugging Face Hub) to check.
        revision (str, optional): The model revision (branch or commit hash). Defaults to "main".

    Raises:
        ValueError: If the model has an unsupported version.
    """
    version = get_base_model_version(diffusers_model_name, revision, local_files_only)
    if version not in allowed_versions:
        raise ValueError(
            f"Model '{diffusers_model_name}' (revision='{revision}') has an unsupported version: '{version.name}'. "
            f"Supported versions: {[v.name for v in allowed_versions]}."
        )

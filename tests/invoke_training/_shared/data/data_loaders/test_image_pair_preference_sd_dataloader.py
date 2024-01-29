import torch

from invoke_training._shared.data.data_loaders.image_pair_preference_sd_dataloader import (
    build_image_pair_preference_sd_dataloader,
)
from invoke_training.pipelines._experimental.sd_dpo_lora.config import (
    HFHubImagePairPreferenceDatasetConfig,
    ImagePairPreferenceSDDataLoaderConfig,
)


def test_build_image_pair_preference_sd_dataloader():
    """Smoke test of build_image_pair_preference_sd_dataloader(...)."""

    config = ImagePairPreferenceSDDataLoaderConfig(dataset=HFHubImagePairPreferenceDatasetConfig())
    data_loader = build_image_pair_preference_sd_dataloader(config, 4)

    example = next(iter(data_loader))
    assert set(example.keys()) == {
        "id",
        "image_0",
        "original_size_hw_0",
        "crop_top_left_yx_0",
        "prefer_0",
        "image_1",
        "original_size_hw_1",
        "crop_top_left_yx_1",
        "prefer_1",
        "caption",
    }

    for image_key in ["image_0", "image_1"]:
        image = example[image_key]
        assert image.shape == (4, 3, 512, 512)
        assert image.dtype == torch.float32

    assert len(example["caption"]) == 4

    for orig_size_key in ["original_size_hw_0", "original_size_hw_1"]:
        original_size_hw = example[orig_size_key]
        assert len(original_size_hw) == 4
        assert len(original_size_hw[0]) == 2

    for crop_key in ["crop_top_left_yx_0", "crop_top_left_yx_1"]:
        crop_top_left_yx = example[crop_key]
        assert len(crop_top_left_yx) == 4
        assert len(crop_top_left_yx[0]) == 2

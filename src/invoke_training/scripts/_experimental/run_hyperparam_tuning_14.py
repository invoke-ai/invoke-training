import os

from invoke_training.pipelines.invoke_train import train
from invoke_training.scripts._experimental.presets.config_presets import (
    AnyFieldOverride,
    BaseOutputDirOverride,
    PipelineConfigOverride,
    get_sdxl_lora_preset_config,
)


def run_training(
    run_name: str,
    jsonl_path: str,
    dataset_size: int,
    validation_prompts: list,
    overrides: list[PipelineConfigOverride] = None,
):
    base_output_dir = os.path.join("output/hp_tuning/sdxl_lora/", run_name)

    overrides = overrides or []
    all_overrides = overrides + [
        BaseOutputDirOverride(base_output_dir=base_output_dir),
    ]

    train_config = get_sdxl_lora_preset_config(
        jsonl_path=jsonl_path, dataset_size=dataset_size, validation_prompts=validation_prompts, overrides=all_overrides
    )

    train(train_config)


def main():
    tests = [
        {
            "run_name": "red_long",
            "jsonl_path": "/home/ubuntu/data/red_sketches_color_8/metadata.jsonl",
            "dataset_size": 79,
            "validation_prompts": [
                "A digital concept sketch of a red character, front view, white background, a cyberpunk woman wearing "
                "black spandex and golden armor, wearing a black hood her face is hidden",
                "A digital concept sketch of a red character, back view, white background, a muscular warrior wearing "
                "skeletal armor, his face is masked and his arms are chained",
            ],
            "overrides": [AnyFieldOverride("data_loader.dataset.image_column", "file_name")],
        },
    ]

    for test in tests:
        run_training(**test)


if __name__ == "__main__":
    main()

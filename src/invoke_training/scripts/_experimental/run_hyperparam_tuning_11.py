import os

from invoke_training.pipelines.invoke_train import train
from invoke_training.scripts._experimental.presets.config_presets import (
    BaseOutputDirOverride,
    PipelineConfigOverride,
    get_sd_lora_preset_config,
)


def run_training(
    run_name: str,
    jsonl_path: str,
    dataset_size: int,
    validation_prompts: list,
    overrides: list[PipelineConfigOverride] = None,
):
    base_output_dir = os.path.join("output/hp_tuning/sd_lora/", run_name)

    overrides = overrides or []
    all_overrides = overrides + [
        BaseOutputDirOverride(base_output_dir=base_output_dir),
    ]

    train_config = get_sd_lora_preset_config(
        jsonl_path=jsonl_path, dataset_size=dataset_size, validation_prompts=validation_prompts, overrides=all_overrides
    )

    train(train_config)


def main():
    tests = [
        {
            "run_name": "yeti",
            "jsonl_path": "/home/ryan/data/yeti/output.jsonl",
            "dataset_size": 29,
            "validation_prompts": [
                "a product photo of a black YT1Cool on a white background",
                "a white YT1Cool on the beach",
            ],
        },
    ]

    for test in tests:
        run_training(**test)


if __name__ == "__main__":
    main()

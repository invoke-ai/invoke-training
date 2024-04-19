import os

from invoke_training.pipelines.invoke_train import train
from invoke_training.scripts._experimental.config_presets import (
    AnyFieldOverride,
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
            "run_name": "baroque",
            "jsonl_path": "data/nga-baroque/metadata.jsonl",
            "dataset_size": 381,
            "validation_prompts": [
                "A baroque painting of a woman carrying a basket of fruit.",
                "A baroque painting of a cute Yoda creature.",
            ],
            "overrides": [
                AnyFieldOverride("data_loader.caption_prefix", "A baroque painting of"),
            ],
        },
        {
            "run_name": "yeti",
            "jsonl_path": "/home/ryan/data/yeti/output.jsonl",
            "dataset_size": 29,
            "validation_prompts": [
                "a product photo of a black YT1Cool on a white background",
                "a white YT1Cool on the beach",
            ],
        },
        {
            "run_name": "face",
            "jsonl_path": "/home/ryan/data/ryan_db_v7/data.jsonl",
            "dataset_size": 13,
            "validation_prompts": [
                "A ohwx young man wearing a Hawaiian shirt a the beach.",
                "A professional headhsot of an ohwx young man wearing a suit. Close up.",
            ],
        },
    ]

    for test in tests:
        run_training(**test)


if __name__ == "__main__":
    main()

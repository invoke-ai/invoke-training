import os

from invoke_training.pipelines.invoke_train import train
from invoke_training.scripts._experimental.config_presets import (
    AnyFieldOverride,
    BaseOutputDirOverride,
    PipelineConfigOverride,
    get_sdxl_ti_preset_config,
)


def run_training(
    run_name: str,
    jsonl_path: str,
    dataset_size: int,
    placeholder_token: str,
    initializer_token: str,
    validation_prompts: list[str],
    caption_preset: str,
    overrides: list[PipelineConfigOverride] = None,
):
    base_output_dir = os.path.join("output/hp_tuning/sdxl_ti/", run_name)

    overrides = overrides or []
    all_overrides = overrides + [
        BaseOutputDirOverride(base_output_dir=base_output_dir),
    ]

    train_config = get_sdxl_ti_preset_config(
        jsonl_path=jsonl_path,
        dataset_size=dataset_size,
        placeholder_token=placeholder_token,
        initializer_token=initializer_token,
        validation_prompts=validation_prompts,
        caption_preset=caption_preset,
        overrides=all_overrides,
    )

    train(train_config)


def main():
    tests = [
        {
            "run_name": "bruce_lr_1e-3",
            "jsonl_path": "/home/ryan/src/invoke-training/sample_data/bruce_the_gnome/bruce_ti.jsonl",
            "dataset_size": 4,
            "placeholder_token": "bruce_the_gnome",
            "initializer_token": "gnome",
            "validation_prompts": [
                "A bruce_the_gnome stuffed gnome sitting on the beach with a pina colada in its hand.",
                "A bruce_the_gnome stuffed gnome reading a book.",
            ],
            "caption_preset": "object",
            "overrides": [
                # AnyFieldOverride("data_loader.dataset.image_column", "file_name"),
                AnyFieldOverride("optimizer.learning_rate", 1e-3),
                # AnyFieldOverride("data_loader.dataset.keep_in_memory", True),
            ],
        },
        {
            "run_name": "face_lr_1e-3",
            "jsonl_path": "/home/ryan/data/ryan_db_v7/data_ti.jsonl",
            "dataset_size": 13,
            "placeholder_token": "ryans_face",
            "initializer_token": "man",
            "validation_prompts": [
                "A ryans_face young man wearing a Hawaiian shirt at the beach.",
                "A professional headshot photo of a ryans_face young man wearing a suit. Grey background.",
            ],
            "caption_preset": "object",
            "overrides": [
                # AnyFieldOverride("data_loader.dataset.image_column", "file_name"),
                AnyFieldOverride("optimizer.learning_rate", 1e-3),
            ],
        },
    ]

    for test in tests:
        run_training(**test)


if __name__ == "__main__":
    main()

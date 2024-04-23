import os

from invoke_training.pipelines.invoke_train import train
from invoke_training.scripts._experimental.presets.config_presets import (
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
            "run_name": "bruce_lr_2e-3",
            "jsonl_path": "/home/ubuntu/src/invoke-training/sample_data/bruce_the_gnome/bruce_ti.jsonl",
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
                AnyFieldOverride("optimizer.learning_rate", 2e-3),
                # AnyFieldOverride("data_loader.dataset.keep_in_memory", True),
            ],
        },
        {
            "run_name": "bruce_lr_4e-3",
            "jsonl_path": "/home/ubuntu/src/invoke-training/sample_data/bruce_the_gnome/bruce_ti.jsonl",
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
                AnyFieldOverride("optimizer.learning_rate", 4e-3),
                # AnyFieldOverride("data_loader.dataset.keep_in_memory", True),
            ],
        },
        {
            "run_name": "mohrbacher_lr_2e-3_vectors_4",
            "jsonl_path": "/home/ubuntu/data/hf_mohrbacher/metadata.jsonl",
            "dataset_size": 180,
            "placeholder_token": "mohrbacher_ti",
            "initializer_token": "mohrbacher",
            "validation_prompts": [
                "in the style of mohrbacher_ti, a complex surreal digital painting of a man wearing a suit, with round "
                "halo above his head, surrounded by animals, by mohrbacher",
                "in the style of mohrbacher_ti, a complex surreal digital painting of a woman with an abstract "
                "geometric head in a red armored dress with a castle in the background by mohrbacher",
            ],
            "caption_preset": "style",
            "overrides": [
                AnyFieldOverride("data_loader.dataset.image_column", "file_name"),
                AnyFieldOverride("optimizer.learning_rate", 2e-3),
                AnyFieldOverride("num_vectors", 4),
            ],
        },
        {
            "run_name": "mohrbacher_lr_2e-3_vectors_16",
            "jsonl_path": "/home/ubuntu/data/hf_mohrbacher/metadata.jsonl",
            "dataset_size": 180,
            "placeholder_token": "mohrbacher_ti",
            "initializer_token": "mohrbacher",
            "validation_prompts": [
                "in the style of mohrbacher_ti, a complex surreal digital painting of a man wearing a suit, with round "
                "halo above his head, surrounded by animals, by mohrbacher",
                "in the style of mohrbacher_ti, a complex surreal digital painting of a woman with an abstract "
                "geometric head in a red armored dress with a castle in the background by mohrbacher",
            ],
            "caption_preset": "style",
            "overrides": [
                AnyFieldOverride("data_loader.dataset.image_column", "file_name"),
                AnyFieldOverride("optimizer.learning_rate", 2e-3),
                AnyFieldOverride("num_vectors", 16),
            ],
        },
    ]

    for test in tests:
        run_training(**test)


if __name__ == "__main__":
    main()

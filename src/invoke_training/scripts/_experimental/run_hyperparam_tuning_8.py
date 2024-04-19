import os

from invoke_training.pipelines.invoke_train import train
from invoke_training.scripts._experimental.config_presets import (
    AnyFieldOverride,
    BaseOutputDirOverride,
    TrainBatchSizeOverride,
    get_sdxl_lora_preset_config,
)


def run_training(base_output_dir: str, jsonl_path: str, dataset_size: int, batch_size: int):
    base_output_dir = os.path.join(base_output_dir, f"bs_{batch_size}")

    overrides = [
        AnyFieldOverride("data_loader.dataset.image_column", "file_name"),  # This is a quirk if the red dataset.
        BaseOutputDirOverride(base_output_dir=base_output_dir),
        TrainBatchSizeOverride(train_batch_size=batch_size),
    ]

    validation_prompts = [
        "A digital concept sketch of a red character, front view, white background, a cyberpunk woman wearing black "
        "spandex and golden armor, wearing a black hood her face is hidden",
        "A digital concept sketch of a red character, back view, white background, a muscular warrior wearing "
        "skeletal armor, his face is masked and his arms are chained",
    ]

    train_config = get_sdxl_lora_preset_config(
        jsonl_path=jsonl_path, dataset_size=dataset_size, validation_prompts=validation_prompts, overrides=overrides
    )

    print(f"Max train steps: {train_config.max_train_steps}")
    train(train_config)


def main():
    output_dir_prefix = "output/hp_tuning/sdxl_red_1x24gb/"
    jsonl_path = "/home/ryan/data/red_sketches_color_8/metadata.jsonl"
    dataset_size = 79

    run_training(output_dir_prefix, jsonl_path, dataset_size, batch_size=4)


if __name__ == "__main__":
    main()

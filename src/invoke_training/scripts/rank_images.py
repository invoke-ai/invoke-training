import argparse
import json
import os
from pathlib import Path
from typing import Literal

import gradio as gr
import yaml
from pydantic import TypeAdapter

from invoke_training.config.pipelines.pipeline_config import PipelineConfig
from invoke_training.config.shared.data.dataset_config import ImagePairPreferenceDatasetConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Choose preferences from image pairs.")
    parser.add_argument(
        "-c",
        "--cfg-file",
        type=Path,
        required=True,
        help="Path to the YAML training config file.",
    )

    return parser.parse_args()


def load_image_pair_preference_dataset_metadata(config: ImagePairPreferenceDatasetConfig):
    assert config.type == "IMAGE_PAIR_PREFERENCE_DATASET"

    metadata = []
    with open(os.path.join(config.dataset_dir, "metadata.jsonl")) as f:
        while (line := f.readline()) != "":
            metadata.append(json.loads(line))

    return metadata


def clip(val, min_val, max_val):
    return max(min(val, max_val), min_val)


def main():
    args = parse_args()

    # Load YAML config file.
    with open(args.cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    pipeline_adapter: TypeAdapter[PipelineConfig] = TypeAdapter(PipelineConfig)
    train_config = pipeline_adapter.validate_python(cfg)

    dataset_config = train_config.data_loader.dataset
    metadata = load_image_pair_preference_dataset_metadata(dataset_config)

    def get_img_path(index: int, image_id: Literal["image_0", "image_1"]):
        return os.path.join(dataset_config.dataset_dir, metadata[index][image_id])

    def get_state(index: int):
        img_0 = get_img_path(index, "image_0")
        img_1 = get_img_path(index, "image_1")
        prefer_0 = metadata[index]["prefer_0"]
        prefer_1 = metadata[index]["prefer_1"]
        return [index, img_0, img_1, prefer_0, prefer_1]

    def go_to_index(index: int):
        new_index = clip(index, 0, len(metadata) - 1)
        return get_state(new_index)

    def mark_prefer_0(index: int):
        metadata[index]["prefer_0"] = True
        metadata[index]["prefer_1"] = False
        # Step to next example.
        return go_to_index(index + 1)

    def mark_prefer_1(index: int):
        metadata[index]["prefer_0"] = False
        metadata[index]["prefer_1"] = True
        # Step to next example.
        return go_to_index(index + 1)

    with gr.Blocks() as demo:
        index = gr.Number(value=0, precision=0)
        with gr.Row():
            img_0 = gr.Image(type="filepath", label="Image 0", interactive=False)
            img_1 = gr.Image(type="filepath", label="Image 1", interactive=False)

        with gr.Row():
            prefer_0 = gr.Checkbox(label="Prefer 0", interactive=False)
            prefer_1 = gr.Checkbox(label="Prefer 1", interactive=False)

        with gr.Row():
            mark_prefer_0_button = gr.Button("Prefer 0")
            mark_prefer_1_button = gr.Button("Prefer 1")

        index.change(go_to_index, inputs=[index], outputs=[index, img_0, img_1, prefer_0, prefer_1])
        mark_prefer_0_button.click(mark_prefer_0, inputs=[index], outputs=[index, img_0, img_1, prefer_0, prefer_1])
        mark_prefer_1_button.click(mark_prefer_1, inputs=[index], outputs=[index, img_0, img_1, prefer_0, prefer_1])

    demo.launch()


if __name__ == "__main__":
    main()

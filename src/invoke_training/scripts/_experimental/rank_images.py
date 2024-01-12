import argparse
import os
import time
from pathlib import Path
from typing import Literal

import gradio as gr
import yaml
from pydantic import TypeAdapter

from invoke_training.config.pipelines.pipeline_config import PipelineConfig
from invoke_training.training.shared.data.datasets.image_pair_preference_dataset import ImagePairPreferenceDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Choose preferences from image pairs.")
    parser.add_argument(
        "-c",
        "--cfg-file",
        type=Path,
        required=True,
        help="Path to the YAML training config file. The internal dataset config will be used.",
    )

    return parser.parse_args()


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
    assert dataset_config.type == "IMAGE_PAIR_PREFERENCE_DATASET"
    metadata = ImagePairPreferenceDataset.load_metadata(dataset_config.dataset_dir)

    print(f"Launching UI to rank image pairs in '{dataset_config.dataset_dir}'.")

    def get_img_path(index: int, image_id: Literal["image_0", "image_1"]):
        return os.path.join(dataset_config.dataset_dir, metadata[index][image_id])

    def get_state(index: int):
        img_0 = get_img_path(index, "image_0")
        img_1 = get_img_path(index, "image_1")
        prefer_0 = metadata[index]["prefer_0"]
        prefer_1 = metadata[index]["prefer_1"]
        caption = metadata[index]["prompt"]
        return [index, img_0, img_1, prefer_0, prefer_1, caption]

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

    def save_metadata():
        timestamp = str(time.time()).replace(".", "_")
        metadata_file = f"metadata-{timestamp}.jsonl"
        metadata_path = ImagePairPreferenceDataset.save_metadata(
            metadata=metadata, dataset_dir=dataset_config.dataset_dir, metadata_file=metadata_file
        )
        print(f"Saved metadata to '{metadata_path}'.")

    with gr.Blocks() as demo:
        index = gr.Number(value=-1, precision=0)
        with gr.Row():
            img_0 = gr.Image(type="filepath", label="Image 0", interactive=False)
            img_1 = gr.Image(type="filepath", label="Image 1", interactive=False)

        caption = gr.Textbox(interactive=False, show_label=False)

        with gr.Row():
            prefer_0 = gr.Checkbox(label="Prefer 0", interactive=False)
            prefer_1 = gr.Checkbox(label="Prefer 1", interactive=False)

        with gr.Row():
            mark_prefer_0_button = gr.Button("Prefer 0")
            mark_prefer_1_button = gr.Button("Prefer 1")

        save_metadata_button = gr.Button("Save Metadata")

        index.change(go_to_index, inputs=[index], outputs=[index, img_0, img_1, prefer_0, prefer_1, caption])
        mark_prefer_0_button.click(
            mark_prefer_0, inputs=[index], outputs=[index, img_0, img_1, prefer_0, prefer_1, caption]
        )
        mark_prefer_1_button.click(
            mark_prefer_1, inputs=[index], outputs=[index, img_0, img_1, prefer_0, prefer_1, caption]
        )
        save_metadata_button.click(save_metadata)

    demo.launch()


if __name__ == "__main__":
    main()

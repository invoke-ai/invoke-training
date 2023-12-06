import argparse
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoImageProcessor, Dinov2Model

from invoke_training.training.tools.cohesion_clustering import (
    choose_best_cluster,
    cluster_images,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetuning with LoRA for Stable Diffusion v1 and v2 base models.")
    parser.add_argument(
        "-i",
        "--in-dir",
        type=Path,
        required=True,
        help="Path to a directory of input images.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        help="Path to the directory of output images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda")
    dtype = torch.float16

    # TODO: Add support for CLIP image encoding.
    # image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.model, torch_dtype=dtype).to(device)
    # image_processor = CLIPImageProcessor()

    # DINOv2
    image_encoder = Dinov2Model.from_pretrained("facebook/dinov2-base")
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    image_encoder.to(device, dtype=dtype)

    in_dir: Path = args.in_dir
    in_paths = [p for p in in_dir.iterdir() if p.suffix in (".jpg", ".jpeg", ".png")]
    clusters = cluster_images(
        in_paths, image_processor, image_encoder, device=device, dtype=dtype, target_cluster_size=7
    )
    best_cluster = choose_best_cluster(clusters, min_cluster_size=6)

    out_dir: Path = args.out_dir
    for cluster in clusters:
        dest_dir = out_dir / str(cluster.label)
        os.makedirs(dest_dir)
        print(
            f"Cluster {cluster.label: >2}: num_images = {len(cluster.image_paths): >3}, "
            f"cohesion distance = {cluster.mean_dist_from_center:.3f}, location = '{dest_dir}'"
        )
        for image_path in cluster.image_paths:
            shutil.copyfile(src=image_path, dst=dest_dir / image_path.name)

    print(f"Most cohesive cluster: '{best_cluster.label}'.")


if __name__ == "__main__":
    main()

import argparse
import os
import shutil
from pathlib import Path

import torch
from transformers import CLIPVisionModelWithProjection

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
    parser.add_argument("-m", "--model", type=Path, required=True, help="Path to the CLIP Vision model.")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda")
    dtype = torch.float16

    clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.model, torch_dtype=dtype).to(device)

    in_dir: Path = args.in_dir
    in_paths = [p for p in in_dir.iterdir() if p.suffix in (".jpg", ".jpeg", ".png")]
    clusters = cluster_images(in_paths, clip_image_encoder, device=device, dtype=dtype, target_cluster_size=10)
    best_cluster = choose_best_cluster(clusters, min_cluster_size=5)

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

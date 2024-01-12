import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from pydantic import TypeAdapter
from torch.utils.data import DataLoader

from invoke_training.config.pipelines.pipeline_config import PipelineConfig
from invoke_training.training._shared.data.data_loaders.dreambooth_sd_dataloader import build_dreambooth_sd_dataloader
from invoke_training.training._shared.data.data_loaders.image_caption_sd_dataloader import (
    build_image_caption_sd_dataloader,
)
from invoke_training.training._shared.data.data_loaders.image_pair_preference_sd_dataloader import (
    build_image_pair_preference_sd_dataloader,
)
from invoke_training.training._shared.data.data_loaders.textual_inversion_sd_dataloader import (
    build_textual_inversion_sd_dataloader,
)


def save_image(torch_image: torch.Tensor, out_path: Path):
    """Save a torch image to disk.

    Args:
        torch_image (torch.Tensor): Shape=(C, H, W). Pixel values are expected to be normalized in the range
            [-1.0, 1.0].
        out_path (Path): The output path.
    """
    np_image = torch_image.clone().detach().cpu().numpy()

    # Convert back to range [0, 1.0].
    np_image = np_image * 0.5 + 0.5
    # Convert back to range [0, 255].
    np_image *= 255
    # Move channel axis from first dimension to last dimension.
    np_image = np.moveaxis(np_image, 0, -1)

    # Cast to np.uint8.
    np_image = np_image.astype(np.uint8)

    Image.fromarray(np_image).save(out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize data loading from a pipeline config.")
    parser.add_argument(
        "-c",
        "--cfg-file",
        type=Path,
        required=True,
        help="Path to the YAML training config file.",
    )

    return parser.parse_args()


def visualize(data_loader: DataLoader):
    out_dir = Path(f"out_{str(time.time()).replace('.', '-')}/")
    os.makedirs(out_dir)

    for batch_idx, batch in enumerate(data_loader):
        print(f"Batch {batch_idx}:")
        batch_path = out_dir / f"batch_{batch_idx}"
        batch_path.mkdir()
        saved_images = []
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: Tensor.shape={v.shape}")
                if len(v.shape) == 4 and v.shape[1] == 3:
                    # This is likely a batch of RGB images, so we save them to disk.
                    for i in range(v.shape[0]):
                        out_path = batch_path / f"{k}_{i}.png"
                        save_image(v[i, ...], out_path)
                        saved_images.append(out_path)
            else:
                print(f"{k}: {v}")

        for saved_image in saved_images:
            print(f"Saved image to '{saved_image}'.")

        _ = input("\n\nPress Enter to continue to next batch...\n")


def main():
    args = parse_args()

    # Load YAML config file.
    with open(args.cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    pipeline_adapter: TypeAdapter[PipelineConfig] = TypeAdapter(PipelineConfig)
    train_config = pipeline_adapter.validate_python(cfg)

    data_loader_config = train_config.data_loader

    if data_loader_config.type == "IMAGE_CAPTION_SD_DATA_LOADER":
        data_loader = build_image_caption_sd_dataloader(
            config=data_loader_config,
            batch_size=train_config.train_batch_size,
            shuffle=False,
        )
    elif data_loader_config.type == "TEXTUAL_INVERSION_SD_DATA_LOADER":
        data_loader = build_textual_inversion_sd_dataloader(
            config=data_loader_config,
            placeholder_str=train_config.placeholder_token,
            batch_size=train_config.train_batch_size,
            shuffle=False,
        )
    elif data_loader_config.type == "DREAMBOOTH_SD_DATA_LOADER":
        data_loader = build_dreambooth_sd_dataloader(
            config=data_loader_config,
            batch_size=train_config.train_batch_size,
            shuffle=False,
            sequential_batching=False,
        )
    elif data_loader_config.type == "IMAGE_PAIR_PREFERENCE_SD_DATA_LOADER":
        data_loader = build_image_pair_preference_sd_dataloader(
            config=data_loader_config,
            batch_size=train_config.train_batch_size,
            shuffle=False,
        )
    else:
        raise ValueError(f"Unexpected data loader type: '{data_loader_config.type}'.")

    visualize(data_loader)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import torch
import torch.utils.data
from tqdm import tqdm

from invoke_training.scripts._experimental.masks.clipseg import load_clipseg_model, run_clipseg, select_device
from invoke_training.scripts.utils.image_dir_dataset import ImageDirDataset, list_collate_fn


@torch.no_grad()
def generate_masks(image_dir: str, prompt: str, clipseg_temp: float, batch_size: int):
    """Generate masks for a directory of images.

    Args:
        image_dir (str): The directory containing images.
        prompt (str): A short description of the thing you want to mask. E.g. 'a cat'.
        clipseg_temp (float): Temperature applied to the CLIPSeg logits. Higher values cause the mask to be 'smoother'.
            and include more of the background. Recommended range: 0.5 to 1.0.
        batch_size (int): Batch size to use when processing images. Larger batch sizes may be faster but require more.
    """
    device = select_device()

    clipseg_processor, clipseg_model = load_clipseg_model()

    # Prepare the dataloader.
    dataset = ImageDirDataset(image_dir)
    print(f"Found {len(dataset)} images in '{image_dir}'.")
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=list_collate_fn, batch_size=batch_size, drop_last=False
    )

    # Process each image.
    for batch in tqdm(data_loader):
        masks = run_clipseg(
            images=batch["image"],
            prompt=prompt,
            clipseg_processor=clipseg_processor,
            clipseg_model=clipseg_model,
            clipseg_temp=clipseg_temp,
            device=device,
        )

        for image_path, mask in zip(batch["image_path"], masks, strict=True):
            image_path = Path(image_path)
            out_path = image_path.parent / "masks" / (image_path.stem + ".png")
            out_path.parent.mkdir(exist_ok=True, parents=True)
            mask.save(out_path)
            print(f"Saved mask to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate masks for a directory of images.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument(
        "--prompt",
        required=True,
        type=str,
        help="A short description of the thing you want to mask. E.g. 'a cat'.",
    )
    parser.add_argument(
        "--clipseg-temp",
        type=float,
        default=1.0,
        help="Temperature applied to the CLIPSeg logits. Higher values cause the mask to be 'smoother' and include "
        "more of the background. Recommended range: 0.5 to 1.0.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size to use when processing images. Larger batch sizes may be faster but require more memory.",
    )
    args = parser.parse_args()

    generate_masks(image_dir=args.dir, prompt=args.prompt, clipseg_temp=args.clipseg_temp, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

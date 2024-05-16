import argparse
from pathlib import Path

import torch
import torch.utils.data
from tqdm import tqdm

from invoke_training._shared.data.datasets.image_caption_jsonl_dataset import (
    MASK_COLUMN_DEFAULT,
    ImageCaptionJsonlDataset,
)
from invoke_training._shared.utils.jsonl import load_jsonl, save_jsonl
from invoke_training.scripts._experimental.masks.clipseg import load_clipseg_model, run_clipseg, select_device


def collate_fn(examples):
    """A collate_fn that combines images into a list rather than stacking into a tensor."""
    return {
        "id": [example["id"] for example in examples],
        "image": [example["image"] for example in examples],
    }


def validate_out_json_path(out_json_path: str | Path):
    out_json_path = Path(out_json_path)
    if out_json_path.exists():
        raise FileExistsError(f"Output jsonl file '{out_json_path}' already exists.")
    if not out_json_path.suffix == ".jsonl":
        raise ValueError(f"Output jsonl file '{out_json_path}' must have a .jsonl extension.")


@torch.no_grad()
def generate_masks(
    in_jsonl_path: str,
    out_jsonl_path: str,
    image_column: str,
    caption_column: str,
    prompt: str,
    clipseg_temp: float,
    batch_size: int,
):
    """Generate masks for a .jsonl dataset."""
    # Load the .jsonl dataset.
    dataset = ImageCaptionJsonlDataset(
        jsonl_path=in_jsonl_path, image_column=image_column, caption_column=caption_column
    )
    print(f"Loaded dataset from '{in_jsonl_path}' with {len(dataset)} images.")
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, drop_last=False)

    # We also need the raw jsonl data.
    jsonl_data = load_jsonl(in_jsonl_path)

    # Prepare output locations.
    out_jsonl_path = Path(out_jsonl_path)
    validate_out_json_path(out_jsonl_path)
    out_masks_dir = out_jsonl_path.parent / "masks"
    out_masks_dir.mkdir(exist_ok=False, parents=True)

    clipseg_processor, clipseg_model = load_clipseg_model()

    device = select_device()

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

        for id, mask in zip(batch["id"], masks, strict=True):
            orig_image_path = Path(jsonl_data[int(id)][image_column])
            out_mask_path: Path = out_masks_dir / (orig_image_path.stem + ".png")
            mask.save(out_mask_path)
            print(f"Saved mask to: {out_mask_path}")

            # Infer whether the mask path should be relative or absolute based on the image path.
            if orig_image_path.is_absolute():
                jsonl_data[int(id)][MASK_COLUMN_DEFAULT] = str(out_mask_path.resolve())
            else:
                jsonl_data[int(id)][MASK_COLUMN_DEFAULT] = str(out_mask_path.relative_to(out_jsonl_path.parent))

    # Save the modified jsonl data.
    validate_out_json_path(out_jsonl_path)
    save_jsonl(jsonl_data, out_jsonl_path)
    print(f"Saved modified jsonl data to: {out_jsonl_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate masks for a jsonl dataset.")
    parser.add_argument("--in-jsonl", type=str, required=True, help="Path to the dataset .jsonl file.")
    parser.add_argument(
        "--out-jsonl",
        type=str,
        required=True,
        help="Path to save the modified .jsonl file to. A masks/ directory will be created in the same directory as "
        "the .jsonl file to store the masks. The choice of whether to use relative or absolute paths for the masks is "
        "inferred from the image paths.",
    )
    parser.add_argument(
        "--image-column",
        type=str,
        default="image",
        help="The name of the column containing image paths in the input .jsonl file.",
    )
    parser.add_argument(
        "--caption-column",
        type=str,
        default="text",
        help="The name of the column containing captions in the input .jsonl file.",
    )
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

    generate_masks(
        in_jsonl_path=args.in_jsonl,
        out_jsonl_path=args.out_jsonl,
        image_column=args.image_column,
        caption_column=args.caption_column,
        prompt=args.prompt,
        clipseg_temp=args.clipseg_temp,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

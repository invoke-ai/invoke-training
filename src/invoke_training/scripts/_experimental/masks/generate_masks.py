import argparse
from pathlib import Path

import torch
import torch.utils.data
from PIL import Image
from transformers import AutoProcessor, CLIPSegForImageSegmentation

from invoke_training.scripts.utils.image_dir_dataset import ImageDirDataset, list_collate_fn


def select_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16

    return torch.device("cpu"), torch.float32


def run_clipseg(
    images: list[Image.Image],
    prompt: str,
    clipseg_processor,
    clipseg_model,
    clipseg_temp: float,
) -> list[Image.Image]:
    """Run ClipSeg on a list of images.

    Args:
        clipseg_temp (float): Temperature applied to the CLIPSeg logits. Higher values cause the mask to be 'smoother'
            and include more of the background. Recommended range: 0.5 to 1.0.
    """

    orig_image_sizes = [img.size for img in images]

    prompts = [prompt] * len(images)
    # TODO(ryand): Should we run the same image with and without the prompt to normalize for any bias in the model?
    inputs = clipseg_processor(text=prompts, images=images, padding=True, return_tensors="pt")
    outputs = clipseg_model(**inputs)

    logits = outputs.logits
    if logits.ndim == 2:
        # The model squeezes the batch dimension if it's 1, so we need to unsqueeze it.
        logits = logits.unsqueeze(0)
    probs = torch.nn.functional.sigmoid(logits / clipseg_temp)
    # Normalize each mask to 0-255. Note that each mask is normalized independently.
    probs = 255 * probs / probs.amax(dim=(1, 2), keepdim=True)

    # Make mask greyscale.
    masks: list[Image.Image] = []
    for prob, orig_size in zip(probs, orig_image_sizes, strict=True):
        mask = Image.fromarray(prob.cpu().numpy()).convert("L")
        mask = mask.resize(orig_size)
        masks.append(mask)

    return masks


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
    device, dtype = select_device_and_dtype()

    # Load the model.
    clipseg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    # Prepare the dataloader.
    dataset = ImageDirDataset(image_dir)
    print(f"Found {len(dataset)} images in '{image_dir}'.")
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=list_collate_fn, batch_size=batch_size, drop_last=False
    )

    # Process each image.
    for batch in data_loader:
        image_path = [0]
        masks = run_clipseg(
            images=batch["image"],
            prompt=prompt,
            clipseg_processor=clipseg_processor,
            clipseg_model=clipseg_model,
            clipseg_temp=clipseg_temp,
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

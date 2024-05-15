import argparse

import torch
import torch.utils.data
from PIL import Image
from transformers import AutoProcessor, CLIPSegForImageSegmentation

from invoke_training.scripts.utils.image_dir_dataset import ImageDirDataset, list_collate_fn


def select_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16

    return torch.device("cpu"), torch.float32


@torch.no_grad()
def generate_masks(image_dir: str, prompt: str, clipseg_temp: float, clipseg_bias: float = 0.01):
    device, dtype = select_device_and_dtype()

    # Load the model.
    clipseg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    # Prepare the dataloader.
    dataset = ImageDirDataset(image_dir)
    print(f"Found {len(dataset)} images in '{image_dir}'.")
    # TODO(ryand): Can we run with a larger batch_size?
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=list_collate_fn, batch_size=1, drop_last=False)

    # Process each image.
    for batch in data_loader:
        image = batch["image"][0]
        original_size = image.size
        image_path = batch["image_path"][0]
        print(f"Processing image: {image_path}")
        # We run the same image with and without the prompt to normalize for any bias in the model.
        inputs = clipseg_processor(text=[prompt, ""], images=[image] * 2, padding=True, return_tensors="pt")
        outputs = clipseg_model(**inputs)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits / clipseg_temp, dim=0)[0]
        probs = (probs + clipseg_bias).clamp_(0, 1)
        probs = 255 * probs / probs.max()

        # Make mask greyscale.
        mask = Image.fromarray(probs.cpu().numpy()).convert("L")

        # Resize mask to original size.
        mask = mask.resize(original_size)

        # TODO(ryand): Save mask to file.
        return


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
        help="Temperature for sampling from the CLIPSeg model. Higher values cause the mask to include more of the "
        "background. Recommended range: 0.5 to 1.0.",
    )
    args = parser.parse_args()

    generate_masks(image_dir=args.dir, prompt=args.prompt, clipseg_temp=args.clipseg_temp)


if __name__ == "__main__":
    main()

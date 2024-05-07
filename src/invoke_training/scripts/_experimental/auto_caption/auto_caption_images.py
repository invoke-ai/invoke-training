import argparse
import json
import os
import typing
from pathlib import Path

import torch
import torch.utils.data
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def collate_fn(examples):
    """Custom collate_fn that combines images into a list rather than stacking into a tensor. This is what moondream
    expects.
    """
    return {
        "image": [example["image"] for example in examples],
        "image_path": [example["image_path"] for example in examples],
    }


class ImageDirDataset(torch.utils.data.Dataset):
    """A simple dataset that loads images from a directory."""

    def __init__(
        self,
        dataset_dir: str,
        image_extensions: typing.Optional[list[str]] = None,
    ):
        super().__init__()
        if image_extensions is None:
            image_extensions = [".png", ".jpg", ".jpeg"]
        image_extensions = [ext.lower() for ext in image_extensions]

        # Determine the list of image paths to include in the dataset.
        self._image_paths: list[str] = []
        for image_file in os.listdir(dataset_dir):
            image_path = os.path.join(dataset_dir, image_file)
            if os.path.isfile(image_path) and os.path.splitext(image_path)[1].lower() in image_extensions:
                self._image_paths.append(image_path)
        self._image_paths.sort()

    def _load_image(self, image_path: str) -> Image.Image:
        # We call `convert("RGB")` to drop the alpha channel from RGBA images, or to repeat channels for greyscale
        # images.
        return Image.open(image_path).convert("RGB")

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int):
        image_path = self._image_paths[idx]
        image = self._load_image(image_path)
        return {"image_path": self._image_paths[idx], "image": image}


def select_device_and_dtype(force_cpu: bool = False) -> tuple[torch.device, torch.dtype]:
    if force_cpu:
        return torch.device("cpu"), torch.float32

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16

    return torch.device("cpu"), torch.float32


def process_images(images: list[Image.Image], prompt: str, moondream, tokenizer) -> list[str]:
    # image_embeds = moondream.encode_image(image).to(device=device)
    # answer = moondream.answer_question(image_embeds, prompt, tokenizer)
    answers = moondream.batch_answer(
        images=images,
        prompts=[prompt] * len(images),
        tokenizer=tokenizer,
    )
    return answers


def main(image_dir: str, prompt: str, use_cpu: bool, batch_size: int):
    device, dtype = select_device_and_dtype(use_cpu)
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Load the model.
    model_id = "vikhyatk/moondream2"
    model_revision = "2024-04-02"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
    # TODO(ryand): Warn about security implications of trust_remote_code=True.
    moondream_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=model_revision
    ).to(device=device, dtype=dtype)
    moondream_model.eval()

    # Prepare the dataloader.
    dataset = ImageDirDataset(image_dir)
    print(f"Found {len(dataset)} images in '{image_dir}'.")
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, drop_last=False)

    results = []
    for image_batch in tqdm(data_loader):
        image_paths = image_batch["image_path"]
        answers = process_images(image_batch["image"], prompt, moondream_model, tokenizer)
        for image_path, answer in zip(image_paths, answers, strict=True):
            results.append({"image": image_path, "text": answer})

    out_path = Path("output.jsonl")
    if out_path.exists():
        raise FileExistsError(f"Output file already exists: {out_path}")

    with open(out_path, "w") as outfile:
        for entry in results:
            json.dump(entry, outfile)
            outfile.write("\n")
    print("Output saved to output.jsonl.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the moondream captioning model on a directory of images.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in 20 words or less.",
        help="(Optional) Prompt for the model.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Force use of CPU instead of GPU. If not set, a GPU will be used if available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing images. To maximize speed, set this to the largest value that fits in GPU "
        "memory.",
    )
    args = parser.parse_args()

    main(args.dir, args.prompt, args.cpu, args.batch_size)

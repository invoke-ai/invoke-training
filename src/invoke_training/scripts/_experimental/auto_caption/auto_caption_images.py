import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


def select_device_and_dtype(force_cpu: bool = False) -> tuple[torch.device, torch.dtype]:
    if force_cpu:
        return torch.device("cpu"), torch.float32

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16

    return torch.device("cpu"), torch.float32


def process_image(image_path: str, prompt: str, moondream, tokenizer, device: torch.device):
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_embeds = moondream.encode_image(image).to(device=device)
    answer = moondream.answer_question(image_embeds, prompt, tokenizer)
    return answer


def main(image_dir: str, prompt: str, use_cpu: bool):
    device, dtype = select_device_and_dtype(use_cpu)
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Load the model.
    model_id = "vikhyatk/moondream2"
    model_revision = "2024-04-02"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
    # TODO(ryand): Warn about security implication of trust_remote_code=True.
    moondream_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=model_revision
    ).to(device=device, dtype=dtype)
    moondream_model.eval()

    results = []
    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            image_path = os.path.join(image_dir, image_file)
            answer = process_image(image_path, prompt, moondream_model, tokenizer, device)
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
        "--prompt", type=str, default="Describe this image in 20 words or less.", help="Prompt for the model."
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Force use of CPU instead of GPU. If not set, a GPU will be used if available.",
    )
    args = parser.parse_args()

    main(args.dir, args.prompt, args.cpu)

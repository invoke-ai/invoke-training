import argparse
import json
import os

import torch
from moondream import LATEST_REVISION, Moondream, detect_device
from PIL import Image
from transformers import AutoTokenizer


def process_image(image_path, prompt, moondream, tokenizer, device):
    print(f"Processing image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        image_embeds = moondream.encode_image(image).to(device=device)
        answer = moondream.answer_question(image_embeds, prompt, tokenizer)
        print(f"Completed: {image_path}")
        return answer
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "Error processing image."


def main(image_folder, prompt, use_cpu):
    device, dtype = ("cpu", torch.float32) if use_cpu else detect_device()
    if not use_cpu:
        print(f"Using device: {device}")

    model_id = "vikhyatk/moondream2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
    moondream = Moondream.from_pretrained(model_id, revision=LATEST_REVISION, torch_dtype=dtype).to(device=device)
    moondream.eval()

    results = []
    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            image_path = os.path.join(image_folder, image_file)
            answer = process_image(image_path, prompt, moondream, tokenizer, device)
            results.append({"image": image_path, "text": answer})

    with open("output.jsonl", "w") as outfile:
        for entry in results:
            json.dump(entry, outfile)
            outfile.write("\n")
    print("Output saved to output.jsonl.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Moondream model on a folder of images.")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing images")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for the model")
    parser.add_argument("--cpu", action="store_true", help="Force use of CPU instead of GPU")
    args = parser.parse_args()

    main(args.folder, args.prompt, args.cpu)

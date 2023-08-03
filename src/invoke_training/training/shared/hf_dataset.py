import os
import random

import datasets
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer

from invoke_training.training.lora.lora_training_config import DatasetConfig


def initialize_hf_dataloader(
    config: DatasetConfig,
    accelerator: Accelerator,
    tokenizer: CLIPTokenizer,
    batch_size: int,
) -> DataLoader:
    """This function was mostly copied from
    https://github.com/huggingface/diffusers/blob/a74c995e7da9cae390ea0918c490f2551a217659/examples/text_to_image/train_text_to_image.py

    TODO(ryand): Review this function, add tests, and tidy it up.
    """
    # In distributed training, the load_dataset function guarantees that only
    # one local process will download the dataset.
    if config.name is not None:
        # Download the dataset from the Hugging Face hub.
        dataset = datasets.load_dataset(
            config.name,
            config.dataset_config_name,
            cache_dir=config.hf_cache_dir,
        )
    elif config.dataset_dir is not None:
        data_files = {}
        data_files["train"] = os.path.join(config.dataset_dir, "**")
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
        dataset = datasets.load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=config.hf_cache_dir,
        )
    else:
        raise ValueError("At least one of 'name' or 'dataset_dir' must be set.")

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # Get the column names for input/target.
    if config.image_column not in column_names:
        raise ValueError(
            f"The image_column='{config.image_column}' is not in the set of dataset column names: '{column_names}'."
        )
    if config.caption_column not in column_names:
        raise ValueError(
            f"The dataset_caption_column='{config.caption_column}' is not in the set of dataset column names: "
            f"'{column_names}'."
        )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[config.caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column '{config.caption_column}' should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                config.resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            (
                transforms.CenterCrop(config.resolution)
                if config.center_crop
                else transforms.RandomCrop(config.resolution)
            ),
            (transforms.RandomHorizontalFlip() if config.random_flip else transforms.Lambda(lambda x: x)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[config.image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=config.dataloader_num_workers,
    )

    return train_dataloader

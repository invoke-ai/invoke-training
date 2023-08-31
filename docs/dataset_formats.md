# Image Caption Dataset Formats
Many of the `invoke-training` training methods require datasets consisting of image-caption pairs. This document explains the supported dataset formats, and how they can be prepared.

`invoke-training` uses [Hugging Face Datasets](https://huggingface.co/docs/datasets/main/en/index) to handle datasets. Image caption datasets can either be downloaded from the [Hugging Face Hub](https://huggingface.co/datasets), or loaded from a directory using the [ImageFolder](https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder) format. More details are included in the following sections.

### Hugging Face Hub Datasets
The easiest way to get started with `invoke-training` is to use a publicly available dataset on [Hugging Face Hub](https://huggingface.co/datasets). You can filter for the `Text-to-Image` task to find relevant datasets that contain both an image column and a caption column. [lambdalabs/pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) is a popular choice if you're not sure where to start.

Once you've selected a dataset from the Hugging Face Hub, you can use it in training by setting the following parameters in your training config:
- `dataset_name`
- `image_column`
- `caption_column`

Here's a sample configuration:
```yaml
dataset:
  dataset_name: lambdalabs/pokemon-blip-captions
  image_column: image
  caption_column: text

# ...
```

See [data_config.py](/src/invoke_training/training/config/data_config.py) for full documentation of the `ImageCaptionDataLoaderConfig`.

### ImageFolder Datasets
If you want to create custom datasets, then you will most likely want to use the [ImageFolder](https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder) dataset format.

An `ImageFolder` dataset should have the following directory structure:
```bash
my_custom_dataset/
├── metadata.jsonl
└── train/
    ├── 0001.png
    ├── 0002.png
    ├── 0003.png
    └── ...
```

The contents of `metadata.jsonl` should be:
```json
{"file_name": "train/0001.png", "text": "This is a caption describing image 0001."}
{"file_name": "train/0002.png", "text": "This is a caption describing image 0002."}
{"file_name": "train/0003.png", "text": "This is a caption describing image 0003."}
```

To use a custom `ImageFolder` dataset, set the following parameters in your training config:
- `dataset_dir`
- `caption_column`

Here's a sample configuration:
```yaml
dataset:
  dataset_dir: /data/my_custom_dataset
  caption_column: text

# ...
```

See [data_config.py](/src/invoke_training/training/config/data_config.py) for full documentation of the `ImageCaptionDataLoaderConfig`.

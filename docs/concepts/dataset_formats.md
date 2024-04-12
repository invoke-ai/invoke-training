# Dataset Formats

`invoke-training` supports the following dataset formats:

- `IMAGE_CAPTION_JSONL_DATASET`: A local image-caption dataset described by a single `.jsonl` file.
- `IMAGE_CAPTION_DIR_DATASET`: A local directory of images with associated `.txt` caption files.
- `IMAGE_DIR_DATASET`: A local directory of images (without captions).
- `HF_HUB_IMAGE_CAPTION_DATASET`: A Hugging Face Hub dataset containing images and captions.

See the documentation for a particular training pipeline to see which dataset formats it supports.

The following sections explain each of these formats in more detail.

## `IMAGE_CAPTION_JSONL_DATASET`

Config documentation: [ImageCaptionJsonlDatasetConfig][invoke_training.config.data.dataset_config.ImageCaptionJsonlDatasetConfig]

A `IMAGE_CAPTION_JSONL_DATASET` consists of a single `.jsonl` file containing image paths and associated captions.

Sample directory structure:
```bash
my_custom_dataset/
├── data.jsonl
└── train/
    ├── 0001.png
    ├── 0002.png
    ├── 0003.png
    └── ...
```

The contents of `data.jsonl` would be:
```json
{"file_name": "train/0001.png", "text": "This is a caption describing image 0001."}
{"file_name": "train/0002.png", "text": "This is a caption describing image 0002."}
{"file_name": "train/0003.png", "text": "This is a caption describing image 0003."}
```

The image file paths can be either absolute paths, or relative to the `.jsonl` file.

Finally, this dataset can be used with the following pipeline dataset configuration:
```yaml
type: IMAGE_CAPTION_JSONL_DATASET
jsonl_path: /path/to/my_custom_dataset/metadata.jsonl
image_column: file_name
caption_column: text
```

A useful characteristic of this dataset format is that a `.jsonl` file can reference an image file anywhere on the local disk. It is common to maintain multiple `.jsonl` datasets that reference some of the same images without needing multiple copies of those images on disk.

## `IMAGE_CAPTION_DIR_DATASET`

Config documentation: [ImageCaptionDirDataset][invoke_training.config.data.dataset_config.ImageCaptionDirDatasetConfig]

A `IMAGE_CAPTION_DIR_DATASET` consists of a directory of image files and corresponding `.txt` caption files of the same name.

Sample directory structure:
```bash
my_custom_dataset/
├── 0001.png
├── 0001.txt
├── 0002.jpg
├── 0002.txt
├── 0003.png
├── 0003.txt
└── ...
```

Each `.txt` file should contain a caption on the first line of the file. Here are the sample contents of `0001.txt`:
```txt title="0001.txt"
this is a caption for example 0001
```

This dataset can be used with the following pipeline dataset configuration:
```yaml
type: IMAGE_CAPTION_DIR_DATASET
dataset_dir: /path/to/my_custom_dataset
```

## `IMAGE_DIR_DATASET`

Config documentation: [ImageDirDataset][invoke_training.config.data.dataset_config.ImageDirDatasetConfig]

A `IMAGE_DIR_DATASET` consists of a single directory of images (without captions).

Sample directory structure:
```bash
my_custom_dataset/
├── 0001.png
├── 0002.jpg
├── 0003.png
└── ...
```

This dataset can be used with the following pipeline dataset configuration:
```yaml
type: IMAGE_DIR_DATASET
dataset_dir: /path/to/my_custom_dataset
```

## `HF_HUB_IMAGE_CAPTION_DATASET`

Config documentation: [HFHubImageCaptionDatasetConfig][invoke_training.config.data.dataset_config.HFHubImageCaptionDatasetConfig]

The easiest way to get started with `invoke-training` is to use a publicly available dataset on [Hugging Face Hub](https://huggingface.co/datasets). You can filter for the `Text-to-Image` task to find relevant datasets that contain both an image column and a caption column. [lambdalabs/pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) is a popular choice if you're not sure where to start.

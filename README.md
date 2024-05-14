# invoke-training

A library for training custom Stable Diffusion models (fine-tuning, LoRA training, textual inversion, etc.) that can be used in [InvokeAI](https://github.com/invoke-ai/InvokeAI).

> [!WARNING]
> `invoke-training` is still under active development, and breaking changes are likely. Full backwards compatibility will not be guaranteed until v1.0.0.
> In the meantime, I recommend pinning to a specific commit hash.

## Documentation

https://invoke-ai.github.io/invoke-training/

## Training Modes

- Stable Diffusion
  - LoRA
  - DreamBooth LoRA
  - Textual Inversion
- Stable Diffusion XL
  - Full finetuning
  - LoRA
  - DreamBooth LoRA
  - Textual Inversion
  - LoRA and Textual Inversion

More training modes coming soon!

## Installation

For more installation details, see the [Installation](https://invoke-ai.github.io/invoke-training/get-started/installation/) section of the documentation.

```bash
# A recent version of pip is required, so first upgrade pip:
python -m pip install --upgrade pip

# Editable install:
pip install -e ".[test]" --extra-index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

`invoke-training` pipelines can be configured and launched from either the CLI or the GUI.

### CLI

Run training via the CLI with type-checked YAML configuration files for maximum control:

```bash
invoke-train --cfg-file src/invoke_training/sample_configs/sdxl_textual_inversion_gnome_1x24gb.yaml
```

### GUI

Run training via the GUI for a simpler starting point.

```bash
invoke-train-ui

# Or, you can optionally override the default host and port:
invoke-train-ui --host 0.0.0.0 --port 1234
```

## Features

Training progress can be monitored with [Tensorboard](https://www.tensorflow.org/tensorboard):
![Screenshot of the Tensorboard UI showing validation images.](docs/images/tensorboard_val_images_screenshot.png)
_Validation images in the Tensorboard UI._

All trained models are compatible with InvokeAI:

![Screenshot of the InvokeAI UI with an example of a Yoda pokemon generated using a Pokemon LoRA model.](docs/images/invokeai_yoda_pokemon_lora.png)
_Example image generated with the prompt "A cute yoda pokemon creature." and a trained Pokemon LoRA._

## Contributing

Contributors are welcome. For developer guidance, see the [Contributing](https://invoke-ai.github.io/invoke-training/contributing/development_environment/) section of the documentation.

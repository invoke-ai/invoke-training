# Quick Start

`invoke-training` has both a GUI and a CLI (for advanced users). The instructions for getting started with both options can be found on this page.

There is also a video introduction to `invoke-training`:

<iframe width="560" height="315" src="https://www.youtube.com/embed/OZIz2vvtlM4?si=iR73F0IhlsolyYAl" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Quick Start - GUI

### 1. Installation

Follow the [`invoke-training` installation instructions](./installation.md).

### 2. Launch the GUI

```bash
# From the invoke-training directory:
invoke-train-ui

# Or, you can optionally override the default host and port:
invoke-train-ui --host 0.0.0.0 --port 1234
```

Access the GUI in your browser at the URL printed to the console.

### 3. Configure the training job

Select the desired training pipeline type in the top-level tab.

For this tutorial, we don't need to change any of the configuration values. The preset configuration should work well.

### 4. Generate the YAML configuration

Click on 'Generate Config' to generate a YAML configuration file. This YAML configuration file could be used to launch the training job from the CLI, if desired.

### 5. Start training

Click on the 'Start Training' and check your terminal for progress logs.

### 6. Monitor training

Monitor the training process with Tensorboard by running `tensorboard --logdir output/` and visiting [localhost:6006](http://localhost:6006) in your browser. Here you can see generated validation images throughout the training process.

![Screenshot of the Tensorboard UI showing validation images.](../images/tensorboard_val_images_screenshot.png)
_Validation images in the Tensorboard UI._

### 7. Invokeai

Select a checkpoint based on the quality of the generated images.

If you haven't already, setup [InvokeAI](https://github.com/invoke-ai/InvokeAI) by following its documentation.

Copy your selected LoRA checkpoint into your `${INVOKEAI_ROOT}/autoimport/lora` directory. For example:

```bash
# Note: You will have to replace the timestamp in the checkpoint path.
cp output/1691088769.5694647/checkpoint_epoch-00000002.safetensors ${INVOKEAI_ROOT}/autoimport/lora/pokemon_epoch-00000002.safetensors
```

You can now use your trained Pokemon LoRA in the InvokeAI UI! 🎉

![Screenshot of the InvokeAI UI with an example of a Yoda pokemon generated using a Pokemon LoRA model.](../images/invokeai_yoda_pokemon_lora.png)
_Example image generated with the prompt "A cute yoda pokemon creature." and Pokemon LoRA._

## Quick Start - CLI

### 1. Installation

Follow the [`invoke-training` installation instructions](./installation.md).

### 2. Training

See the [Textual Inversion - SDXL](../guides/stable_diffusion/textual_inversion_sdxl.md) tutorial for instructions on how to train a model via the CLI.

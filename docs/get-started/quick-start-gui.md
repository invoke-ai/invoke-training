# Quick Start - GUI

This page walks through the steps to train your first model with the `invoke-training` GUI.

There is also a [Quick Start - CLI](./quick-start-cli.md) guide.

## Tutorial

### 1. Installation
Follow the [`invoke-training` installation instructions](./installation.md).

### 2. Launch the GUI
```bash
# From the invoke-training directory:
invoke-train-ui
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
*Validation images in the Tensorboard UI.*

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
*Example image generated with the prompt "A cute yoda pokemon creature." and Pokemon LoRA.*

## Next Steps

After completing this Quick Start tutorial, we recommend continuing with any of the [full training pipeline tutorials](../tutorials/index.md).

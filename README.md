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

This project supports **Python 3.10, 3.11, and 3.12**.

**Important:** This project requires PyTorch. For optimal performance, especially when training on GPUs, installing the correct PyTorch version compatible with your hardware (CUDA or ROCm) is crucial.

---

### Method 1: Manual Installation (Recommended)

This method gives you the most control over ensuring the correct PyTorch version is installed for your system.

**1. Set up a Python Virtual Environment:**

Using a virtual environment is strongly recommended to avoid dependency conflicts.

```bash
# Create a virtual environment (e.g., named .venv)
python -m venv .venv

# Activate the environment
# Windows (Command Prompt/PowerShell):
.venv\Scripts\activate
# Linux/macOS (bash/zsh):
source .venv/bin/activate
```

**2. Install PyTorch:**

Visit the [official PyTorch installation page](https://pytorch.org/get-started/locally/) and follow the instructions to install the version appropriate for your operating system, package manager (pip), and compute platform (CUDA version, ROCm version, or CPU).

*   **Example (CUDA 12.4):** If you have CUDA 12.4 compatible drivers, the command might be:
    ```bash
    # Ensure pip is up-to-date
    pip install --upgrade pip
    # Install PyTorch with CUDA 12.4 support
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu124
    ```
*   **Example (CPU Only):**
    ```bash
    # Ensure pip is up-to-date
    pip install --upgrade pip
    # Install PyTorch for CPU
    pip install torch    ```

*Make sure you run this command while your virtual environment is active.*

**3. Install `invoke-training`:**

Once PyTorch is installed correctly, install this package from the project's root directory:

```bash
# Ensure your virtual environment is active
pip install .

# Optional: For development (editable install with test dependencies)
# pip install -e ".[test]"
```

---

### Method 2: Using Setup Scripts (Beta)

We provide experimental menu-driven scripts (`setup_and_run.sh` for Linux/macOS, `setup_and_run.bat` for Windows) to simplify the setup and running process. These scripts attempt to automate virtual environment creation, PyTorch installation (defaulting to CUDA 12.4), dependency installation, and running the application.

**Note:** These scripts are currently in **beta**. The automatic PyTorch installation might not work for all systems (especially if you need a different CUDA version or CPU-only). If you encounter issues, please use the Manual Installation method above.

**How to Use:**

1.  Navigate to the root directory of this project in your terminal.
2.  Run the appropriate script for your operating system:
    *   **Linux/macOS:**
        ```bash
        # Make the script executable (only needed once)
        chmod +x setup_and_run.sh
        # Run the script
        ./setup_and_run.sh
        ```
    *   **Windows:**
        ```batch
        # Run the script (double-clicking might also work)
        setup_and_run.bat
        ```
3.  Follow the on-screen menu prompts:
    *   **Install:** Creates a `.venv` virtual environment, attempts to install PyTorch (confirming the default CUDA 12.4 version first), and installs `invoke-training`. If you decline the default PyTorch install, you will need to install it manually (see Method 1, Step 2) and then potentially re-run the install option.
    *   **Run:** Activates the existing `.venv` environment and starts the `invoke-train-ui`.
    *   **Reinstall:** Deletes the existing `.venv` directory and runs the installation process again.
    *   **Exit:** Closes the script.

Even when using the scripts, you will still need to activate the virtual environment manually (`source .venv/bin/activate` or `.venv\Scripts\activate.bat`) if you want to run commands like `invoke-train` directly from your terminal outside the script's Run option.

---

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

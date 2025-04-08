# Installation

## Requirements

1. Python 3.10, 3.11 and 3.12 are currently supported. Check your Python version by running `python -V`.
2. An NVIDIA GPU with >= 8 GB VRAM is recommended for model training.

## Basic Installation

0. Open your terminal and navigate to the directory where you want to clone the `invoke-training` repo.
1. Clone the repo:

   ```bash
   git clone https://github.com/invoke-ai/invoke-training.git
   ```

2. Create and activate a python [virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments). This creates an isolated environment for `invoke-training` and its dependencies that won't interfere with other python environments on your system, including any installations of [InvokeAI](https://www.github.com/invoke-ai/invokeai).

   ```bash
   # Navigate to the invoke-training directory.
   cd invoke-training

   # Create a new virtual environment named `invoketraining`.
   python -m venv invoketraining

   # Activate the new virtual environment.
   # On Windows:
   .\invoketraining\Scripts\activate
   # On MacOS / Linux:
   source invoketraining/bin/activate
   ```

3. Install `invoke-training` and its dependencies. Run the appropriate install command for your system.

   ```bash
   # A recent version of pip is required, so first upgrade pip:
   python -m pip install --upgrade pip

   # Install - Windows or Linux with a Nvidia GPU:
   pip install ".[test]" --extra-index-url https://download.pytorch.org/whl/cu126

   # Install - Linux with no GPU:
   pip install ".[test]" --extra-index-url https://download.pytorch.org/whl/cpu

   # Install - All other systems:
   pip install ".[test]"
   ```

## Developer Installation

Consider forking the repo if you plan to contribute code changes.

Follow the above installation instructions, cloning your fork instead of this repo if you made a fork.

Next, we suggest setting up the repo's pre-commit hooks to automatically format and lint your contributions:

1. (_Optional_) Install the pre-commit hooks: `pre-commit install`. This will run static analysis tools (ruff) on `git commit`.
2. (_Optional_) Setup `ruff` in your IDE of choice.

# Installation

## Requirements

1. Python `>= 3.10` is currently supported. Check your Python version by running `python -V`.
2. An NVIDIA GPU with >= 8 GB VRAM is recommended for model training.

## Basic Installation
1. Clone the repo:
```bash
git clone https://github.com/invoke-ai/invoke-training.git
```
2. (*Optional, but highly recommended*) Create and activate a python [virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments). This creates an isolated environment for `invoke-training` and its dependencies that won't interfere with other python environments on your system, including any installations of the [local Invoke client](https://www.github.com/invoke-ai/invokeai).
```bash
# Create the new virtual environment in a memorable location by navigating to the folder and running this command
python -m venv invoketraining

# Activate the new virtual environment
Windows:  .\invoketraining\Scripts\activate
Linux: source invoketraining/bin/activate

```
3. Install `invoke-training` and its dependencies:
```bash
# A recent version of pip is required, so first upgrade pip:
python -m pip install --upgrade pip

# Install:
pip install ".[test]" --extra-index-url https://download.pytorch.org/whl/cu121
```

## Developer Installation
1. Consider forking the repo if you plan to contribute code changes.
2. `git clone` the repo.
3. (*Optional, but highly recommended*) Create and activate a python [virtual environment](https://docs.python.org/3/library/venv.html#creating-virtual-environments). This creates an isolated environment for `invoke-training` and its dependencies that won't interfere with other python environments on your system, including any installations of the [local Invoke client](https://www.github.com/invoke-ai/invokeai).
```bash
# Create the new virtual environment in a memorable location by navigating to the folder and running this command
python -m venv invoketraining

# Activate the new virtual environment
Windows:  .\invoketraining\Scripts\activate
Linux: source invoketraining/bin/activate

```
4. Install `invoke-training` and its dependencies:
```bash
# A recent version of pip is required, so first upgrade pip:
python -m pip install --upgrade pip

# Editable install:
pip install -e ".[test]" --extra-index-url https://download.pytorch.org/whl/cu121
```
5. (*Optional*) Install the pre-commit hooks: `pre-commit install`. This will run static analysis tools (ruff) on `git commit`.
6. (*Optional*) Setup `ruff` in your IDE of choice.

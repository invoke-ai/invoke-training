# Installation

## Basic Installation
1. Clone the repo:
```bash
git clone https://github.com/invoke-ai/invoke-training.git
```
2. (Optional, but recommended) Create a python virtual environment.
3. Install `invoke-training` and its dependencies:
```bash
# A recent version of pip is required, so first upgrade pip:
python -m pip install --upgrade pip

# Install:
pip install ".[test]" --extra-index-url https://download.pytorch.org/whl/cu118
```
4. Test out your installation by following the 

## Developer Installation
1. Consider forking the repo if you plan to contribute code changes.
2. `git clone` the repo.
3. (Optional, but recommended) Create a python virtual environment.
4. Install `invoke-training` and its dependencies:
```bash
# A recent version of pip is required, so first upgrade pip:
python -m pip install --upgrade pip

# Editable install:
pip install -e ".[test]" --extra-index-url https://download.pytorch.org/whl/cu118
```
5. (Optional) Install the pre-commit hooks: `pre-commit install`. This will run static analysis tools (ruff) on `git commit`.
6. (Optional) Setup `ruff` in your IDE of choice.

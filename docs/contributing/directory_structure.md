# Directory Structure

```bash
invoke-training/
├── README.md
├── docs/
├── src/
│   └── invoke-training/
│       ├── _shared/ # Utilities shared across multiple pipelines. Hight unit test coverage.
│       ├── config/ # Config structures shared by multiple pipelines.
│       ├── pipelines/ # Each pipeline is isolated in it's own directory with a train.py and config.py.
│       │   ├── stable_diffusion/
│       │   │   ├── lora/
│       │   │   │   ├── config.py
│       │   │   │   └── train.py
│       │   │   └── textual_inversion/
│       │   │       └── ...
│       │   ├── stable_diffusion_xl/
│       │   └── ...
│       └── scripts/ # Main entrypoints.
└── tests/ # Mirrors src/ directory.
```

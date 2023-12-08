# Directory Structure

```bash
invoke-training/
├── README.md
├── docs/
├── src/
│   └── invoke-training/
│       ├── scripts/ # Main entrypoints.
│       ├── core/ # Shared across multiple pipelines. High unit test coverage. Likely to be imported by downstream libraries.
│       │   ├── lora
│       │   └── ...
│       ├── training/
│       │   ├── shared/
│       │   │   ├── checkpoints # Shared utils related to checkpointing.
│       │   │   ├── data # Data handling that is shared across multiple pipelines
│       │   │   ├── stable_diffusion # Things that are shared by many SD pipelines.
│       │   │   ├── stable_diffusion_lora # Things that are shared by multiple SD LoRA pipelines.
│       │   │   └── ...
│       │   └── pipelines/
│       │       ├── stable_diffusion/
│       │       │   ├── train_lora.py
│       │       │   ├── train_textual_inversion.py
│       │       │   └── train_pivotal_tuning.py
│       │       └── stable_diffusion_xl/
│       │           ├── train_lora.py
│       │           ├── train_textual_inversion.py
│       │           └── train_pivotal_tuning.py
│       └── config/ # Mirrors training/ structure.
└── tests/ # Mirrors src/ directory.
```
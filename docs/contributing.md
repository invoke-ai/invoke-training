# Directory Structure

```bash
invoke-training/
├── README.md
├── docs/
├── src/
│   └── invoke-training/
│       ├── scripts/ # Main entrypoints.
│       ├── core/ # Utils shared across multiple pipelines. High unit test coverage. Likely to be imported by downstream libraries.
│       │   ├── lora
│       │   ├── utils
│       │   └── ...
│       ├── training/
│       │   ├── shared/ # Training utils shared by multiple pipelines.
│       │   │   ├── data # Data handling that is shared across multiple pipelines.
│       │   │   ├── stable_diffusion_1_or_2 # Things that are shared across SD1 or 2 pipelines.
│       │   │   ├── lora # Things that are shared across lora pipelines.
│       │   │   └── ...
│       │   └── pipelines/
│       │       ├── stable_diffusion_1_or_2/
│       │       │   ├── lora/
│       │       │   │   └── train.py
│       │       │   ├── textual_inversion/
│       │       │   │   └── train.py
│       │       │   ├── pivotal_tuning/
│       │       │   │   └── train.py
│       │       │   └── ...
│       │       └── stable_diffusion_xl/
│       │           ├── lora/
│       │           │   └── train.py
│       │           ├── textual_inversion/
│       │           │   └── train.py
│       │           ├── pivotal_tuning/
│       │           │   └── train.py
│       │           └── ...
│       └── config/ # Mirrors pipelines/ structure.
└── tests/ # Mirrors src/ directory.
```
# Training Worker with FLUX Support

A distributed AI model training worker that supports both FLUX and SDXL model fine-tuning with LoRA, Textual Inversion, and full fine-tuning capabilities.

## Features

### Supported Training Types
- **FLUX LoRA** ✨ **NEW**: High-quality FLUX.1-dev LoRA training
- **SDXL LoRA**: Stable Diffusion XL LoRA training
- **SDXL Textual Inversion**: Custom token training for SDXL
- **SDXL Finetune**: Full model fine-tuning for SDXL

### Key Capabilities
- **Memory-Optimized**: Multiple GPU memory tier support (24GB/40GB/80GB+)
- **Commercial Presets**: Production-ready configurations for cost optimization
- **Hardware Validation**: Automatic checks for training feasibility
- **Cost Estimation**: Built-in cost estimation for training jobs
- **Flexible Configuration**: Easy-to-use presets with full customization options

## Installation

```bash
# Clone or copy the training-worker directory
cd training-worker

# Install dependencies
pip install -e .

# Or install from requirements if you prefer
pip install -r requirements.txt
```

## Quick Start

### Command Line Usage

```bash
# Check supported training types and hardware requirements
python -m training_worker.training_worker --list-types

# Get hardware information
python -m training_worker.training_worker --hardware-info

# Run a training job from config file
python -m training_worker.training_worker --job-config examples/flux_lora_job.json
```

### Python API Usage

```python
from training_worker.wrapper import TrainingWorkerAPI, flux_lora_job, sdxl_lora_job

# Initialize the API
api = TrainingWorkerAPI()

# Check what training types are supported
print("Supported types:", api.get_supported_training_types())

# Check hardware feasibility
feasibility = api.check_training_feasibility("FLUX_LORA")
print("FLUX feasible:", feasibility["valid"])

# Create and run a FLUX LoRA job
job_config = flux_lora_job(
    job_id="my_flux_job",
    dataset_path="/path/to/my/dataset",
    output_path="/path/to/output",
    gpu_memory_tier="40gb",
    cost_optimization="balanced",
    max_train_steps=1000
)

result = api.submit_job(job_config)
print("Training result:", result["status"])
```

## FLUX Training

### Hardware Requirements

| GPU Memory | Status | Batch Size | Training Speed | Cost/Hour |
|------------|--------|------------|----------------|-----------|
| 24GB+ | Minimum | 1 | Slow | ~$2.50 |
| 40GB+ | Recommended | 2 | Good | ~$3.50 |
| 80GB+ | Optimal | 4 | Fast | ~$7.00 |

### Memory Optimization Strategies

**24GB GPU (RTX 4090, A5000)**:
- Batch size: 1, Gradient accumulation: 8
- Enable VAE and text encoder caching
- Enable CPU offload during validation
- Use bfloat16 precision

**40GB GPU (A100)**:
- Batch size: 2, Gradient accumulation: 4  
- Enable VAE and text encoder caching
- Disable CPU offload
- Use bfloat16 precision

**80GB+ GPU (H100)**:
- Batch size: 4, Gradient accumulation: 2
- Disable caching (not needed)
- Use float16 for best quality
- Disable gradient checkpointing

### FLUX Configuration Examples

```python
# Quick FLUX job with automatic optimization
job = flux_lora_job(
    job_id="flux_portrait_training",
    dataset_path="/datasets/portraits", 
    output_path="/models/flux_portraits",
    gpu_memory_tier="40gb",  # Automatically optimizes for your GPU
    cost_optimization="balanced",  # or "aggressive" or "quality"
    max_train_steps=1000,
    resolution=768
)

# Advanced FLUX configuration
job = api.create_simple_job(
    job_id="advanced_flux",
    training_type="FLUX_LORA",
    dataset_path="/datasets/custom",
    output_path="/models/custom_flux",
    resolution=1024,
    batch_size=1,
    learning_rate=1e-4,
    lora_rank=32,  # Higher rank = more expressive
    max_train_steps=2000,
    config_overrides={
        "timestep_sampler": "shift",  # FLUX-specific
        "discrete_flow_shift": 3.0,   # FLUX-specific
        "train_text_encoder": False,  # Very memory intensive
        "cache_vae_outputs": True,
        "gradient_checkpointing": True,
        "validation_prompts": [
            "A detailed portrait of a person",
            "A beautiful landscape at sunset"
        ]
    }
)
```

## SDXL Training

### SDXL Configuration Examples

```python
# Quick SDXL LoRA job
job = sdxl_lora_job(
    job_id="sdxl_style_training",
    dataset_path="/datasets/art_style",
    output_path="/models/sdxl_style",
    cost_optimization="balanced",
    max_train_steps=1000,
    resolution=1024
)

# SDXL Textual Inversion
job = api.create_simple_job(
    job_id="sdxl_ti",
    training_type="SDXL_TEXTUAL_INVERSION",
    dataset_path="/datasets/concept",
    output_path="/models/ti_concept",
    config_overrides={
        "placeholder_token": "<my_concept>",
        "learning_rate": 5e-4
    }
)
```

## Cost Optimization

### Budget-Based Recommendations

```python
# Get recommendations for a $10 budget
recommendations = api.get_preset_recommendations(
    training_type="FLUX_LORA",
    budget_usd=10.0,
    quality_preference="balanced"
)

print("Recommended presets:")
for rec in recommendations["recommendations"]:
    preset = rec["preset_name"] 
    cost = rec["cost_estimate"]["estimated_cost_usd"]
    hours = rec["cost_estimate"]["training_hours"]
    print(f"  {preset}: ${cost:.2f} ({hours:.1f} hours)")
```

### Cost Estimation

```python
# Estimate cost before training
cost_estimate = api.estimate_job_cost(
    training_type="FLUX_LORA",
    max_train_steps=1000,
    resolution=768,
    batch_size=1,
    gpu_cost_per_hour=3.50
)

print(f"Estimated cost: ${cost_estimate['estimated_cost_usd']:.2f}")
print(f"Training time: {cost_estimate['training_hours']:.1f} hours")
print(f"Speed: {cost_estimate['steps_per_minute']:.1f} steps/min")
```

## Commercial Presets

Pre-configured settings optimized for production use:

### FLUX Presets
- `flux_24gb_aggressive`: Fast training on 24GB GPU
- `flux_24gb_balanced`: Balanced quality/speed on 24GB GPU  
- `flux_40gb_aggressive`: Fast training on 40GB GPU
- `flux_40gb_balanced`: Balanced quality/speed on 40GB GPU
- `flux_40gb_quality`: High quality training on 40GB GPU
- `flux_80gb_balanced`: Balanced training on 80GB GPU
- `flux_80gb_quality`: Maximum quality training on 80GB GPU

### SDXL Presets
- `sdxl_aggressive`: Fast, low-cost SDXL training
- `sdxl_balanced`: Balanced SDXL training
- `sdxl_quality`: High-quality SDXL training

```python
from training_worker.config_presets.commercial_config import get_commercial_preset

# Use a commercial preset
config = get_commercial_preset("flux_40gb_balanced")
```

## Dataset Format

Datasets should follow this structure:

```
dataset/
├── data.jsonl          # Metadata file
├── image_001.jpg       # Training images
├── image_002.jpg
└── ...
```

`data.jsonl` format:
```json
{"image": "image_001.jpg", "caption": "A detailed description of the image"}
{"image": "image_002.jpg", "caption": "Another detailed description"}
```

## Configuration Files

### JSON Configuration

```json
{
  "job_id": "my_training_job",
  "training_type": "FLUX_LORA",
  "dataset_path": "/path/to/dataset",
  "output_path": "/path/to/output",
  "resolution": 768,
  "batch_size": 1,
  "learning_rate": 0.0001,
  "max_train_steps": 1000,
  "lora_rank": 16,
  "config_overrides": {
    "gradient_accumulation_steps": 4,
    "cache_vae_outputs": true,
    "validation_prompts": ["A test prompt"]
  }
}
```

### YAML Configuration

```yaml
job_id: my_training_job
training_type: FLUX_LORA
dataset_path: /path/to/dataset
output_path: /path/to/output
resolution: 768
batch_size: 1
learning_rate: 0.0001
max_train_steps: 1000
lora_rank: 16
config_overrides:
  gradient_accumulation_steps: 4
  cache_vae_outputs: true
  validation_prompts:
    - "A test prompt"
```

## Performance Tips

### FLUX Training Optimization

1. **Use appropriate memory tier**: Match your GPU memory to the preset
2. **Enable caching**: VAE and text encoder caching saves significant memory
3. **Gradient accumulation**: Use higher values for smaller batch sizes
4. **Resolution**: Start with 768, increase to 1024 if memory allows
5. **LoRA rank**: Use 16-32 for good quality/size balance

### SDXL Training Optimization

1. **Batch size**: Use 2-4 for most GPUs
2. **Resolution**: 1024 is standard, 512 for faster training
3. **LoRA rank**: 4-16 works well for most use cases
4. **Learning rate**: 1e-4 to 5e-4 depending on dataset size

## Monitoring and Logging

The training worker provides comprehensive logging:

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Training progress is logged automatically
# Check training_worker.log for detailed logs
```

## Troubleshooting

### Common Issues

**Out of Memory Errors (FLUX)**:
- Reduce batch size to 1
- Enable all caching options
- Use gradient checkpointing
- Enable CPU offload during validation

**Slow Training (FLUX)**:
- Use higher GPU memory tier
- Disable unnecessary caching on high-memory GPUs
- Increase batch size if memory allows

**Poor Results**:
- Increase training steps
- Use higher LoRA rank
- Check dataset quality and captions
- Adjust learning rate

### Hardware Compatibility

```python
# Check if your hardware can run FLUX
feasibility = api.check_training_feasibility("FLUX_LORA")
if not feasibility["valid"]:
    print("FLUX not feasible:", feasibility["message"])
    for rec in feasibility["recommendations"]:
        print(f"  - {rec}")
```

## API Reference

### TrainingWorkerAPI

Main API class for interacting with the training worker.

**Methods**:
- `get_supported_training_types()`: List supported training types
- `get_hardware_info()`: Get hardware information
- `check_training_feasibility(training_type)`: Check feasibility
- `estimate_job_cost(...)`: Estimate training costs
- `create_simple_job(...)`: Create job configuration
- `submit_job(job_config)`: Submit job for training
- `quick_train(...)`: Create and submit job in one call

### Convenience Functions

- `flux_lora_job(...)`: Create FLUX LoRA job with sensible defaults
- `sdxl_lora_job(...)`: Create SDXL LoRA job with sensible defaults
- `create_job_from_json_file(path)`: Load job from JSON file
- `save_job_to_json_file(job, path)`: Save job to JSON file

## License

Apache-2.0 License

## Contributing

Contributions are welcome! Please ensure all changes maintain compatibility with both FLUX and SDXL training pipelines.
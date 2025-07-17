# FLUX Integration Plan for Training-Worker

## Overview

The invoke-training library has recently added comprehensive FLUX LoRA training support via [PR #153](https://github.com/invoke-ai/invoke-training/pull/153). This document outlines the necessary changes to integrate this support into the training-worker and the frontend configuration options that should be exposed to users.

## FLUX Support in invoke-training

### Available Training Types
The invoke-training library now supports:
1. **SDXL LoRA** - Existing support
2. **SDXL Textual Inversion** - Existing support  
3. **SDXL Finetune** - Existing support
4. **FLUX LoRA** - **NEW** ✨

### FLUX LoRA Configuration

The `FluxLoraConfig` class provides the following key parameters:

#### Core Model Settings
- `model`: Base model path (default: "black-forest-labs/FLUX.1-dev")
- `transformer_path`: Custom transformer .safetensors file path
- `text_encoder_1_path`: Custom CLIP text encoder path
- `text_encoder_2_path`: Custom T5 text encoder path

#### LoRA-Specific Settings
- `lora_rank_dim`: LoRA rank dimension (default: 4)
- `lora_checkpoint_format`: "invoke_peft" or "kohya" (default: "kohya")
- `flux_lora_target_modules`: Target modules for LoRA layers
- `text_encoder_lora_target_modules`: Text encoder LoRA targets
- `lora_scale`: Scale parameter for LoRA layers

#### Training Configuration
- `train_transformer`: Whether to train the transformer (default: True)
- `train_text_encoder`: Whether to train text encoder (default: False)
- `transformer_learning_rate`: Transformer-specific learning rate
- `text_encoder_learning_rate`: Text encoder-specific learning rate
- `lr_scheduler`: Learning rate scheduler type
- `lr_warmup_steps`: Warmup steps
- `min_snr_gamma`: Min-SNR weighting parameter

#### Memory & Performance
- `weight_dtype`: "float32", "float16", "bfloat16" (default: "float16")
- `mixed_precision`: "no", "fp16", "bf16", "fp8" (default: "no")
- `gradient_checkpointing`: Memory optimization (default: False)
- `gradient_accumulation_steps`: Gradient accumulation (default: 1)
- `cache_text_encoder_outputs`: Cache text encoder outputs for efficiency
- `cache_vae_outputs`: Cache VAE outputs for efficiency

#### FLUX-Specific Parameters
- `timestep_sampler`: "shift" or "uniform" (default: "shift")
- `discrete_flow_shift`: Shift parameter for discrete flow (default: 3.0)
- `sigmoid_scale`: Scale for sigmoid function (default: 1.0)
- `guidance_scale`: Guidance scale for FLUX model (default: 1.0)
- `clip_tokenizer_max_length`: CLIP tokenizer max length (default: 77)
- `t5_tokenizer_max_length`: T5 tokenizer max length (default: 512)

#### Data Loading
- `data_loader.type`: Must be "IMAGE_CAPTION_FLUX_DATA_LOADER"
- `resolution`: Training resolution (e.g., 768)
- `aspect_ratio_buckets`: Aspect ratio bucketing configuration
- `use_masks`: Whether to use image masks for weighted loss

## Training-Worker Integration Plan

### 1. Update Dependencies

First, ensure the training-worker's `pyproject.toml` or `requirements.txt` includes the latest version of invoke-training that contains FLUX support.

```toml
# In pyproject.toml
invoke-training = "^2.0.0"  # Update to version with FLUX support
```

### 2. Update Config Presets

The training-worker should be updated to support FLUX LoRA configurations. Based on the current structure, you'll need to:

#### A. Add FLUX LoRA Config Preset

Create a new config preset in `training_worker/config_presets/config_presets.py`:

```python
def get_flux_lora_config(
    model: str = "black-forest-labs/FLUX.1-dev",
    resolution: int = 768,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    lora_rank: int = 4,
    max_train_steps: int = 1000,
    **kwargs
) -> dict:
    """Generate FLUX LoRA training configuration."""
    
    config = {
        "type": "FLUX_LORA",
        "model": model,
        "lora_rank_dim": lora_rank,
        "train_batch_size": batch_size,
        "max_train_steps": max_train_steps,
        "transformer_learning_rate": learning_rate,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 10,
        "weight_dtype": "bfloat16",
        "gradient_checkpointing": True,
        "lora_checkpoint_format": "kohya",
        "timestep_sampler": "shift",
        "discrete_flow_shift": 3.0,
        "data_loader": {
            "type": "IMAGE_CAPTION_FLUX_DATA_LOADER",
            "resolution": resolution,
            "aspect_ratio_buckets": {
                "target_resolution": resolution,
                "start_dim": resolution // 2,
                "end_dim": resolution * 2,
                "divisible_by": 128
            },
            "dataloader_num_workers": 4
        },
        "optimizer": {
            "optimizer_type": "AdamW",
            "learning_rate": learning_rate
        }
    }
    
    # Apply any additional kwargs
    config.update(kwargs)
    return config
```

#### B. Update Config Registry

Add FLUX LoRA to the supported training types:

```python
SUPPORTED_TRAINING_TYPES = {
    "SDXL_LORA": get_sdxl_lora_config,
    "SDXL_TEXTUAL_INVERSION": get_sdxl_ti_config,
    "SDXL_FINETUNE": get_sdxl_finetune_config,
    "FLUX_LORA": get_flux_lora_config,  # NEW
}
```

### 3. Update Training Worker Logic

The main training worker (`training_worker.py`) should be updated to handle FLUX LoRA jobs:

```python
def process_training_job(job_config: dict):
    """Process a training job based on its configuration."""
    
    training_type = job_config.get("type")
    
    if training_type == "FLUX_LORA":
        return process_flux_lora_job(job_config)
    elif training_type in ["SDXL_LORA", "SDXL_TEXTUAL_INVERSION", "SDXL_FINETUNE"]:
        return process_sdxl_job(job_config)
    else:
        raise ValueError(f"Unsupported training type: {training_type}")

def process_flux_lora_job(job_config: dict):
    """Process a FLUX LoRA training job."""
    from invoke_training.pipelines.flux.lora.config import FluxLoraConfig
    from invoke_training.pipelines.flux.lora.train import train
    
    # Create config object
    config = FluxLoraConfig(**job_config)
    
    # Run training
    train(config)
    
    return {"status": "completed", "type": "FLUX_LORA"}
```

### 4. Update Resource Requirements

FLUX models are significantly larger than SDXL models. Update resource allocation:

```python
RESOURCE_REQUIREMENTS = {
    "SDXL_LORA": {"gpu_memory": "8GB", "system_memory": "16GB"},
    "SDXL_TEXTUAL_INVERSION": {"gpu_memory": "6GB", "system_memory": "12GB"},
    "SDXL_FINETUNE": {"gpu_memory": "24GB", "system_memory": "32GB"},
    "FLUX_LORA": {"gpu_memory": "40GB", "system_memory": "64GB"},  # NEW - Much higher requirements
}
```

## Frontend Configuration Options

### Essential FLUX Configuration Options for UI

The frontend should expose these key configuration options to users:

#### 1. **Model Selection**
```typescript
interface ModelConfig {
  trainingType: "SDXL_LORA" | "SDXL_TEXTUAL_INVERSION" | "SDXL_FINETUNE" | "FLUX_LORA";
  baseModel: string; // For FLUX: "black-forest-labs/FLUX.1-dev"
}
```

#### 2. **Basic Training Parameters**
```typescript
interface BasicTrainingConfig {
  resolution: 512 | 768 | 1024; // FLUX typically uses 768 or 1024
  batchSize: number; // Usually 1-4 for FLUX due to memory requirements
  maxTrainSteps: number;
  learningRate: number; // Default: 1e-4 for FLUX
  loraRank: number; // Default: 4, higher = more expressive but larger
}
```

#### 3. **Advanced FLUX-Specific Options**
```typescript
interface FluxAdvancedConfig {
  // LoRA Configuration
  loraCheckpointFormat: "kohya" | "invoke_peft";
  trainTransformer: boolean; // Default: true
  trainTextEncoder: boolean; // Default: false (memory intensive)
  
  // Performance & Memory
  weightDtype: "float32" | "float16" | "bfloat16";
  gradientCheckpointing: boolean; // Recommended: true for FLUX
  cacheTextEncoderOutputs: boolean; // Memory optimization
  cacheVaeOutputs: boolean; // Memory optimization
  
  // FLUX-Specific
  timestepSampler: "shift" | "uniform";
  discreteFlowShift: number; // Default: 3.0
  guidanceScale: number; // Default: 1.0
  
  // Learning Rate Schedule
  lrScheduler: "linear" | "cosine" | "constant" | "constant_with_warmup";
  lrWarmupSteps: number;
}
```

#### 4. **GPU/Memory Tier Presets**

Given FLUX's high memory requirements, provide preset configurations:

```typescript
interface MemoryPresets {
  "flux_1x24gb": {
    batchSize: 1,
    gradientAccumulation: 4,
    weightDtype: "bfloat16",
    gradientCheckpointing: true,
    cacheVaeOutputs: true
  },
  "flux_1x40gb": {
    batchSize: 2,
    gradientAccumulation: 2,
    weightDtype: "bfloat16", 
    gradientCheckpointing: true,
    cacheVaeOutputs: false
  },
  "flux_1x80gb": {
    batchSize: 4,
    gradientAccumulation: 1,
    weightDtype: "float16",
    gradientCheckpointing: false,
    cacheVaeOutputs: false
  }
}
```

### Recommended UI Flow

1. **Training Type Selection**: Add "FLUX LoRA" as a new option alongside existing SDXL options
2. **Model Selection**: Auto-populate with "black-forest-labs/FLUX.1-dev" for FLUX
3. **Memory/GPU Preset**: Let users select based on their hardware
4. **Basic Parameters**: Resolution, batch size, steps, learning rate
5. **Advanced Options**: Collapsible section with FLUX-specific parameters
6. **Dataset Upload**: Same as existing flow, but validate for FLUX compatibility

### UI Validation & Warnings

- **Memory Warning**: Show warning for FLUX requiring 24GB+ VRAM
- **Resolution Validation**: FLUX works best with 768+ resolution
- **Batch Size Limits**: Recommend batch size 1-2 for most hardware
- **Estimated Training Time**: FLUX training is significantly slower than SDXL

## Migration Checklist

- [ ] Update invoke-training dependency to latest version
- [ ] Add FLUX LoRA config preset to training-worker
- [ ] Update training job processing logic
- [ ] Add FLUX resource requirements
- [ ] Update frontend to include FLUX LoRA option
- [ ] Add FLUX-specific UI configuration options
- [ ] Implement memory preset recommendations
- [ ] Add validation and warnings for FLUX requirements
- [ ] Test end-to-end FLUX training workflow
- [ ] Update documentation for FLUX support

## Hardware Recommendations

### Minimum Requirements for FLUX LoRA
- **GPU**: 24GB VRAM (RTX 4090, A5000, etc.)
- **System RAM**: 32GB+
- **Storage**: 100GB+ free space for model and checkpoints

### Recommended Requirements
- **GPU**: 40GB+ VRAM (A100, H100)
- **System RAM**: 64GB+
- **Storage**: 200GB+ SSD storage

### Performance Expectations
- **Training Speed**: ~2-3x slower than SDXL LoRA
- **Model Size**: ~1-2GB for typical LoRA ranks
- **Training Time**: 1-4 hours for 1000 steps (depending on hardware)
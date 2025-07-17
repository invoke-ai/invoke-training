"""
Configuration presets for different training types including FLUX LoRA support.
"""

from typing import Any, Dict, Optional, Union


def get_sdxl_lora_config(
    model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    resolution: int = 1024,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    lora_rank: int = 4,
    max_train_steps: int = 1000,
    **kwargs
) -> Dict[str, Any]:
    """Generate SDXL LoRA training configuration."""
    
    config = {
        "type": "SDXL_LORA",
        "model": model,
        "lora_rank_dim": lora_rank,
        "train_batch_size": batch_size,
        "max_train_steps": max_train_steps,
        "unet_learning_rate": learning_rate,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 10,
        "weight_dtype": "bfloat16",
        "gradient_checkpointing": True,
        "lora_checkpoint_format": "kohya",
        "data_loader": {
            "type": "IMAGE_CAPTION_SD_DATA_LOADER",
            "resolution": resolution,
            "aspect_ratio_buckets": {
                "target_resolution": resolution,
                "start_dim": resolution // 2,
                "end_dim": resolution * 2,
                "divisible_by": 64
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


def get_sdxl_textual_inversion_config(
    model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    resolution: int = 1024,
    batch_size: int = 1,
    learning_rate: float = 5e-4,
    max_train_steps: int = 1000,
    placeholder_token: str = "<new_concept>",
    **kwargs
) -> Dict[str, Any]:
    """Generate SDXL Textual Inversion training configuration."""
    
    config = {
        "type": "SDXL_TEXTUAL_INVERSION",
        "model": model,
        "train_batch_size": batch_size,
        "max_train_steps": max_train_steps,
        "placeholder_token": placeholder_token,
        "learning_rate": learning_rate,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 10,
        "weight_dtype": "bfloat16",
        "gradient_checkpointing": True,
        "data_loader": {
            "type": "TEXTUAL_INVERSION_SD_DATA_LOADER", 
            "resolution": resolution,
            "aspect_ratio_buckets": {
                "target_resolution": resolution,
                "start_dim": resolution // 2,
                "end_dim": resolution * 2,
                "divisible_by": 64
            },
            "dataloader_num_workers": 4
        },
        "optimizer": {
            "optimizer_type": "AdamW",
            "learning_rate": learning_rate
        }
    }
    
    config.update(kwargs)
    return config


def get_sdxl_finetune_config(
    model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    resolution: int = 1024,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    max_train_steps: int = 1000,
    **kwargs
) -> Dict[str, Any]:
    """Generate SDXL Finetune training configuration."""
    
    config = {
        "type": "SDXL_FINETUNE",
        "model": model,
        "train_batch_size": batch_size,
        "max_train_steps": max_train_steps,
        "unet_learning_rate": learning_rate,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 10,
        "weight_dtype": "bfloat16",
        "gradient_checkpointing": True,
        "data_loader": {
            "type": "IMAGE_CAPTION_SD_DATA_LOADER",
            "resolution": resolution,
            "aspect_ratio_buckets": {
                "target_resolution": resolution,
                "start_dim": resolution // 2,
                "end_dim": resolution * 2,
                "divisible_by": 64
            },
            "dataloader_num_workers": 4
        },
        "optimizer": {
            "optimizer_type": "AdamW",
            "learning_rate": learning_rate
        }
    }
    
    config.update(kwargs)
    return config


def get_flux_lora_config(
    model: str = "black-forest-labs/FLUX.1-dev",
    resolution: int = 768,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    lora_rank: int = 4,
    max_train_steps: int = 1000,
    transformer_learning_rate: Optional[float] = None,
    text_encoder_learning_rate: Optional[float] = None,
    train_text_encoder: bool = False,
    gradient_accumulation_steps: int = 4,
    cache_text_encoder_outputs: bool = True,
    cache_vae_outputs: bool = True,
    timestep_sampler: str = "shift",
    discrete_flow_shift: float = 3.0,
    guidance_scale: float = 1.0,
    memory_preset: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate FLUX LoRA training configuration.
    
    Args:
        model: Base FLUX model path
        resolution: Training resolution (recommended: 768 or 1024)
        batch_size: Training batch size (keep low for FLUX due to memory requirements)
        learning_rate: Base learning rate
        lora_rank: LoRA rank dimension (4-32, higher = more expressive but larger)
        max_train_steps: Maximum training steps
        transformer_learning_rate: Specific LR for transformer (defaults to learning_rate)
        text_encoder_learning_rate: Specific LR for text encoder
        train_text_encoder: Whether to train text encoder (very memory intensive)
        gradient_accumulation_steps: Steps to accumulate gradients (helps with small batch sizes)
        cache_text_encoder_outputs: Cache text encoder outputs for memory efficiency
        cache_vae_outputs: Cache VAE outputs for memory efficiency
        timestep_sampler: "shift" or "uniform" sampling strategy
        discrete_flow_shift: Shift parameter for discrete flow
        guidance_scale: Guidance scale for FLUX model
        memory_preset: Pre-configured memory settings ("24gb", "40gb", "80gb")
        **kwargs: Additional configuration overrides
    """
    
    # Set default learning rates
    if transformer_learning_rate is None:
        transformer_learning_rate = learning_rate
    if text_encoder_learning_rate is None:
        text_encoder_learning_rate = learning_rate
    
    # Apply memory presets
    if memory_preset:
        memory_configs = get_flux_memory_presets()
        if memory_preset in memory_configs:
            preset_config = memory_configs[memory_preset]
            # Override with preset values if not explicitly set
            if "batch_size" not in kwargs:
                batch_size = preset_config["batch_size"]
            if "gradient_accumulation_steps" not in kwargs:
                gradient_accumulation_steps = preset_config["gradient_accumulation_steps"]
            if "cache_vae_outputs" not in kwargs:
                cache_vae_outputs = preset_config["cache_vae_outputs"]
            if "cache_text_encoder_outputs" not in kwargs:
                cache_text_encoder_outputs = preset_config["cache_text_encoder_outputs"]
    
    config = {
        "type": "FLUX_LORA",
        "model": model,
        
        # LoRA Configuration
        "lora_rank_dim": lora_rank,
        "lora_checkpoint_format": "kohya",
        "train_transformer": True,
        "train_text_encoder": train_text_encoder,
        
        # Learning Rates
        "transformer_learning_rate": transformer_learning_rate,
        "text_encoder_learning_rate": text_encoder_learning_rate,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 10,
        
        # Training Configuration
        "train_batch_size": batch_size,
        "max_train_steps": max_train_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        
        # Memory & Performance Optimizations
        "weight_dtype": "bfloat16",
        "mixed_precision": "no",
        "gradient_checkpointing": True,
        "cache_text_encoder_outputs": cache_text_encoder_outputs,
        "cache_vae_outputs": cache_vae_outputs,
        "enable_cpu_offload_during_validation": False,
        
        # FLUX-Specific Parameters
        "timestep_sampler": timestep_sampler,
        "discrete_flow_shift": discrete_flow_shift,
        "sigmoid_scale": 1.0,
        "lora_scale": 1.0,
        "guidance_scale": guidance_scale,
        "clip_tokenizer_max_length": 77,
        "t5_tokenizer_max_length": 512,
        
        # Data Loading (FLUX-specific)
        "data_loader": {
            "type": "IMAGE_CAPTION_FLUX_DATA_LOADER",
            "resolution": resolution,
            "aspect_ratio_buckets": {
                "target_resolution": resolution,
                "start_dim": resolution // 2,
                "end_dim": resolution * 2,
                "divisible_by": 128  # FLUX uses 128 divisibility
            },
            "dataloader_num_workers": 4
        },
        
        # Optimizer
        "optimizer": {
            "optimizer_type": "AdamW",
            "learning_rate": learning_rate
        },
        
        # Validation
        "validation_prompts": [],
        "num_validation_images_per_prompt": 4,
        "use_masks": False,
    }
    
    # Apply any additional kwargs
    config.update(kwargs)
    return config


def get_flux_memory_presets() -> Dict[str, Dict[str, Any]]:
    """
    Get memory-optimized presets for different GPU configurations.
    """
    return {
        "24gb": {
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "cache_vae_outputs": True,
            "cache_text_encoder_outputs": True,
            "gradient_checkpointing": True,
            "weight_dtype": "bfloat16",
            "enable_cpu_offload_during_validation": True,
        },
        "40gb": {
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "cache_vae_outputs": True,
            "cache_text_encoder_outputs": True,
            "gradient_checkpointing": True,
            "weight_dtype": "bfloat16",
            "enable_cpu_offload_during_validation": False,
        },
        "80gb": {
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "cache_vae_outputs": False,
            "cache_text_encoder_outputs": False,
            "gradient_checkpointing": False,
            "weight_dtype": "float16",
            "enable_cpu_offload_during_validation": False,
        }
    }


# Training type registry
SUPPORTED_TRAINING_TYPES = {
    "SDXL_LORA": get_sdxl_lora_config,
    "SDXL_TEXTUAL_INVERSION": get_sdxl_textual_inversion_config,
    "SDXL_FINETUNE": get_sdxl_finetune_config,
    "FLUX_LORA": get_flux_lora_config,  # NEW: FLUX LoRA support
}

# Resource requirements for different training types
RESOURCE_REQUIREMENTS = {
    "SDXL_LORA": {
        "min_gpu_memory_gb": 8,
        "recommended_gpu_memory_gb": 16,
        "min_system_memory_gb": 16,
        "recommended_system_memory_gb": 32,
        "disk_space_gb": 50,
    },
    "SDXL_TEXTUAL_INVERSION": {
        "min_gpu_memory_gb": 6,
        "recommended_gpu_memory_gb": 12,
        "min_system_memory_gb": 12,
        "recommended_system_memory_gb": 24,
        "disk_space_gb": 30,
    },
    "SDXL_FINETUNE": {
        "min_gpu_memory_gb": 24,
        "recommended_gpu_memory_gb": 40,
        "min_system_memory_gb": 32,
        "recommended_system_memory_gb": 64,
        "disk_space_gb": 100,
    },
    "FLUX_LORA": {
        "min_gpu_memory_gb": 24,  # Absolute minimum with heavy optimizations
        "recommended_gpu_memory_gb": 40,  # Comfortable training
        "optimal_gpu_memory_gb": 80,  # Best performance
        "min_system_memory_gb": 32,
        "recommended_system_memory_gb": 64,
        "disk_space_gb": 150,  # FLUX models are large
        "training_speed_multiplier": 0.3,  # FLUX is ~3x slower than SDXL
    }
}


def get_config_for_training_type(
    training_type: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Get configuration for a specific training type with provided parameters.
    
    Args:
        training_type: One of the supported training types
        **kwargs: Configuration parameters to pass to the config function
        
    Returns:
        Complete training configuration dictionary
        
    Raises:
        ValueError: If training_type is not supported
    """
    if training_type not in SUPPORTED_TRAINING_TYPES:
        supported_types = ", ".join(SUPPORTED_TRAINING_TYPES.keys())
        raise ValueError(f"Unsupported training type: {training_type}. Supported types: {supported_types}")
    
    config_func = SUPPORTED_TRAINING_TYPES[training_type]
    return config_func(**kwargs)


def get_resource_requirements(training_type: str) -> Dict[str, Union[int, float]]:
    """
    Get resource requirements for a specific training type.
    
    Args:
        training_type: One of the supported training types
        
    Returns:
        Dictionary with resource requirements
        
    Raises:
        ValueError: If training_type is not supported
    """
    if training_type not in RESOURCE_REQUIREMENTS:
        supported_types = ", ".join(RESOURCE_REQUIREMENTS.keys())
        raise ValueError(f"Unknown training type: {training_type}. Supported types: {supported_types}")
    
    return RESOURCE_REQUIREMENTS[training_type]


def validate_hardware_requirements(training_type: str, available_gpu_memory_gb: float) -> Dict[str, Any]:
    """
    Validate if the available hardware meets the requirements for a training type.
    
    Args:
        training_type: Training type to validate
        available_gpu_memory_gb: Available GPU memory in GB
        
    Returns:
        Dictionary with validation results and recommendations
    """
    requirements = get_resource_requirements(training_type)
    
    min_gpu = requirements["min_gpu_memory_gb"]
    recommended_gpu = requirements["recommended_gpu_memory_gb"]
    
    if available_gpu_memory_gb < min_gpu:
        return {
            "valid": False,
            "level": "insufficient",
            "message": f"Insufficient GPU memory. {training_type} requires at least {min_gpu}GB, but only {available_gpu_memory_gb}GB available.",
            "recommendations": [
                "Consider using a GPU with more memory",
                "Try reducing batch size and enabling all memory optimizations",
                "Consider using gradient checkpointing and caching options"
            ]
        }
    elif available_gpu_memory_gb < recommended_gpu:
        return {
            "valid": True,
            "level": "minimal",
            "message": f"Minimal GPU memory for {training_type}. Training will be slow and require aggressive memory optimizations.",
            "recommendations": [
                "Use batch size 1 with high gradient accumulation",
                "Enable gradient checkpointing",
                "Enable VAE and text encoder output caching",
                "Consider using mixed precision training"
            ]
        }
    else:
        return {
            "valid": True,
            "level": "good",
            "message": f"Sufficient GPU memory for {training_type} training.",
            "recommendations": []
        }
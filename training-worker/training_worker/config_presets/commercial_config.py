"""
Commercial configuration presets optimized for production deployment and cost-effectiveness.
"""

from typing import Any, Dict, Optional

from .config_presets import get_flux_lora_config, get_sdxl_lora_config


def get_commercial_flux_config(
    resolution: int = 768,
    gpu_memory_tier: str = "40gb",
    cost_optimization: str = "balanced",
    **kwargs
) -> Dict[str, Any]:
    """
    Get a commercial FLUX LoRA configuration optimized for production use.
    
    Args:
        resolution: Training resolution (768 or 1024)
        gpu_memory_tier: "24gb", "40gb", or "80gb" GPU memory tier
        cost_optimization: "aggressive", "balanced", or "quality"
        **kwargs: Additional configuration overrides
    
    Returns:
        Optimized FLUX LoRA configuration
    """
    
    # Base configuration based on GPU tier
    if gpu_memory_tier == "24gb":
        base_config = {
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "cache_vae_outputs": True,
            "cache_text_encoder_outputs": True,
            "enable_cpu_offload_during_validation": True,
        }
    elif gpu_memory_tier == "40gb":
        base_config = {
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "cache_vae_outputs": True,
            "cache_text_encoder_outputs": True,
            "enable_cpu_offload_during_validation": False,
        }
    elif gpu_memory_tier == "80gb":
        base_config = {
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "cache_vae_outputs": False,
            "cache_text_encoder_outputs": False,
            "enable_cpu_offload_during_validation": False,
        }
    else:
        raise ValueError(f"Unsupported GPU tier: {gpu_memory_tier}")
    
    # Apply cost optimization strategy
    if cost_optimization == "aggressive":
        # Maximize speed, minimize cost per job
        optimization_config = {
            "max_train_steps": 500,  # Fewer steps for faster completion
            "lora_rank": 8,  # Smaller LoRA for faster training
            "lr_warmup_steps": 5,
            "train_text_encoder": False,  # Skip text encoder training
            "gradient_checkpointing": True,
            "weight_dtype": "bfloat16",
        }
    elif cost_optimization == "balanced":
        # Balance between quality and cost
        optimization_config = {
            "max_train_steps": 1000,
            "lora_rank": 16,
            "lr_warmup_steps": 10,
            "train_text_encoder": False,
            "gradient_checkpointing": True,
            "weight_dtype": "bfloat16",
        }
    elif cost_optimization == "quality":
        # Optimize for best quality
        optimization_config = {
            "max_train_steps": 2000,
            "lora_rank": 32,
            "lr_warmup_steps": 20,
            "train_text_encoder": False,  # Still too expensive for most cases
            "gradient_checkpointing": gpu_memory_tier != "80gb",  # Disable on high-end GPUs
            "weight_dtype": "float16" if gpu_memory_tier == "80gb" else "bfloat16",
        }
    else:
        raise ValueError(f"Unsupported cost optimization: {cost_optimization}")
    
    # Merge configurations
    config_params = {
        "resolution": resolution,
        "memory_preset": gpu_memory_tier,
        **base_config,
        **optimization_config,
        **kwargs  # User overrides take precedence
    }
    
    return get_flux_lora_config(**config_params)


def get_commercial_sdxl_config(
    resolution: int = 1024,
    cost_optimization: str = "balanced",
    **kwargs
) -> Dict[str, Any]:
    """
    Get a commercial SDXL LoRA configuration optimized for production use.
    
    Args:
        resolution: Training resolution (512, 768, or 1024)
        cost_optimization: "aggressive", "balanced", or "quality"
        **kwargs: Additional configuration overrides
    
    Returns:
        Optimized SDXL LoRA configuration
    """
    
    # Apply cost optimization strategy
    if cost_optimization == "aggressive":
        optimization_config = {
            "batch_size": 4,
            "max_train_steps": 500,
            "lora_rank": 4,
            "gradient_checkpointing": True,
            "cache_vae_outputs": True,
        }
    elif cost_optimization == "balanced":
        optimization_config = {
            "batch_size": 2,
            "max_train_steps": 1000,
            "lora_rank": 8,
            "gradient_checkpointing": True,
            "cache_vae_outputs": False,
        }
    elif cost_optimization == "quality":
        optimization_config = {
            "batch_size": 1,
            "max_train_steps": 2000,
            "lora_rank": 16,
            "gradient_checkpointing": False,
            "cache_vae_outputs": False,
        }
    else:
        raise ValueError(f"Unsupported cost optimization: {cost_optimization}")
    
    config_params = {
        "resolution": resolution,
        **optimization_config,
        **kwargs
    }
    
    return get_sdxl_lora_config(**config_params)


# Commercial preset registry
COMMERCIAL_PRESETS = {
    # FLUX presets by GPU tier and optimization
    "flux_24gb_aggressive": lambda **kwargs: get_commercial_flux_config(
        gpu_memory_tier="24gb", cost_optimization="aggressive", **kwargs
    ),
    "flux_24gb_balanced": lambda **kwargs: get_commercial_flux_config(
        gpu_memory_tier="24gb", cost_optimization="balanced", **kwargs
    ),
    "flux_40gb_aggressive": lambda **kwargs: get_commercial_flux_config(
        gpu_memory_tier="40gb", cost_optimization="aggressive", **kwargs
    ),
    "flux_40gb_balanced": lambda **kwargs: get_commercial_flux_config(
        gpu_memory_tier="40gb", cost_optimization="balanced", **kwargs
    ),
    "flux_40gb_quality": lambda **kwargs: get_commercial_flux_config(
        gpu_memory_tier="40gb", cost_optimization="quality", **kwargs
    ),
    "flux_80gb_balanced": lambda **kwargs: get_commercial_flux_config(
        gpu_memory_tier="80gb", cost_optimization="balanced", **kwargs
    ),
    "flux_80gb_quality": lambda **kwargs: get_commercial_flux_config(
        gpu_memory_tier="80gb", cost_optimization="quality", **kwargs
    ),
    
    # SDXL presets by optimization level
    "sdxl_aggressive": lambda **kwargs: get_commercial_sdxl_config(
        cost_optimization="aggressive", **kwargs
    ),
    "sdxl_balanced": lambda **kwargs: get_commercial_sdxl_config(
        cost_optimization="balanced", **kwargs
    ),
    "sdxl_quality": lambda **kwargs: get_commercial_sdxl_config(
        cost_optimization="quality", **kwargs
    ),
}


def get_commercial_preset(preset_name: str, **kwargs) -> Dict[str, Any]:
    """
    Get a commercial configuration preset by name.
    
    Args:
        preset_name: Name of the commercial preset
        **kwargs: Additional configuration overrides
    
    Returns:
        Complete training configuration
    
    Raises:
        ValueError: If preset name is not found
    """
    if preset_name not in COMMERCIAL_PRESETS:
        available_presets = ", ".join(COMMERCIAL_PRESETS.keys())
        raise ValueError(f"Unknown commercial preset: {preset_name}. Available: {available_presets}")
    
    preset_func = COMMERCIAL_PRESETS[preset_name]
    return preset_func(**kwargs)


def estimate_training_cost(
    training_type: str,
    config: Dict[str, Any],
    gpu_cost_per_hour: float = 2.50,  # Example: A100 cost
    overhead_factor: float = 1.2  # 20% overhead for setup/teardown
) -> Dict[str, Any]:
    """
    Estimate the cost of a training job.
    
    Args:
        training_type: Type of training (FLUX_LORA, SDXL_LORA, etc.)
        config: Training configuration
        gpu_cost_per_hour: Cost per GPU hour in USD
        overhead_factor: Multiplier for setup/teardown overhead
    
    Returns:
        Dictionary with cost estimates
    """
    
    # Base estimates for training speed (steps per minute)
    speed_estimates = {
        "FLUX_LORA": {
            "24gb": 0.5,   # Very slow due to memory constraints
            "40gb": 1.0,   # Reasonable speed
            "80gb": 2.0,   # Good speed
        },
        "SDXL_LORA": {
            "8gb": 3.0,
            "16gb": 5.0,
            "24gb": 8.0,
        }
    }
    
    max_steps = config.get("max_train_steps", 1000)
    batch_size = config.get("train_batch_size", 1)
    grad_accumulation = config.get("gradient_accumulation_steps", 1)
    
    # Effective batch size affects training speed
    effective_batch_size = batch_size * grad_accumulation
    
    # Estimate training speed based on type and memory
    if training_type == "FLUX_LORA":
        # Determine memory tier from config
        if config.get("cache_vae_outputs") and config.get("enable_cpu_offload_during_validation"):
            memory_tier = "24gb"
        elif config.get("cache_vae_outputs"):
            memory_tier = "40gb"
        else:
            memory_tier = "80gb"
        
        base_speed = speed_estimates["FLUX_LORA"][memory_tier]
    else:
        # SDXL speed estimation based on batch size
        if effective_batch_size >= 8:
            memory_tier = "24gb"
        elif effective_batch_size >= 4:
            memory_tier = "16gb" 
        else:
            memory_tier = "8gb"
        
        base_speed = speed_estimates.get("SDXL_LORA", {}).get(memory_tier, 3.0)
    
    # Adjust speed based on effective batch size (larger batches are more efficient)
    speed_multiplier = min(2.0, 1.0 + (effective_batch_size - 1) * 0.1)
    adjusted_speed = base_speed * speed_multiplier
    
    # Calculate time estimates
    training_minutes = max_steps / adjusted_speed
    training_hours = training_minutes / 60
    
    # Apply overhead factor
    total_hours = training_hours * overhead_factor
    
    # Calculate costs
    compute_cost = total_hours * gpu_cost_per_hour
    
    return {
        "training_hours": round(training_hours, 2),
        "total_hours_with_overhead": round(total_hours, 2),
        "estimated_cost_usd": round(compute_cost, 2),
        "cost_per_step": round(compute_cost / max_steps, 4),
        "steps_per_minute": round(adjusted_speed, 2),
        "memory_tier": memory_tier,
        "effective_batch_size": effective_batch_size,
    }


def recommend_preset_for_budget(
    training_type: str,
    budget_usd: float,
    quality_preference: str = "balanced"  # "aggressive", "balanced", "quality"
) -> Dict[str, Any]:
    """
    Recommend the best preset for a given budget.
    
    Args:
        training_type: "FLUX_LORA" or "SDXL_LORA"
        budget_usd: Maximum budget in USD
        quality_preference: Quality vs speed preference
    
    Returns:
        Dictionary with recommendations
    """
    recommendations = []
    
    if training_type == "FLUX_LORA":
        preset_options = [
            ("flux_24gb_aggressive", 2.50),
            ("flux_24gb_balanced", 2.50),
            ("flux_40gb_aggressive", 3.50),
            ("flux_40gb_balanced", 3.50),
            ("flux_40gb_quality", 3.50),
            ("flux_80gb_balanced", 7.00),
            ("flux_80gb_quality", 7.00),
        ]
    else:  # SDXL_LORA
        preset_options = [
            ("sdxl_aggressive", 1.50),
            ("sdxl_balanced", 1.50),
            ("sdxl_quality", 1.50),
        ]
    
    for preset_name, gpu_cost_per_hour in preset_options:
        config = get_commercial_preset(preset_name)
        cost_estimate = estimate_training_cost(training_type, config, gpu_cost_per_hour)
        
        if cost_estimate["estimated_cost_usd"] <= budget_usd:
            recommendations.append({
                "preset_name": preset_name,
                "config": config,
                "cost_estimate": cost_estimate,
                "gpu_cost_per_hour": gpu_cost_per_hour,
            })
    
    # Sort by quality preference
    if quality_preference == "aggressive":
        # Prefer speed (lower cost per step)
        recommendations.sort(key=lambda x: x["cost_estimate"]["cost_per_step"])
    elif quality_preference == "quality":
        # Prefer quality (higher steps, better settings)
        recommendations.sort(key=lambda x: -x["config"]["max_train_steps"])
    else:  # balanced
        # Balance cost and quality
        recommendations.sort(key=lambda x: x["cost_estimate"]["estimated_cost_usd"])
    
    return {
        "budget_usd": budget_usd,
        "recommendations": recommendations[:3],  # Top 3 recommendations
        "total_options": len(recommendations)
    }
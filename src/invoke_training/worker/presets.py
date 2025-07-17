"""
Convenience functions for creating common training job configurations.
"""

from typing import Any, Dict, Optional

from .training_worker import TrainingJobConfig
from .worker_api import TrainingWorkerAPI


def flux_lora_job(
    job_id: str,
    dataset_path: str,
    output_path: str,
    gpu_memory_tier: str = "40gb",
    max_train_steps: int = 1000,
    resolution: int = 768,
    lora_rank: int = 16,
    learning_rate: float = 1e-4,
    **kwargs
) -> TrainingJobConfig:
    """
    Create a FLUX LoRA job with sensible defaults.
    
    Args:
        job_id: Unique job identifier
        dataset_path: Path to dataset directory
        output_path: Path for training outputs
        gpu_memory_tier: "24gb", "40gb", or "80gb"
        max_train_steps: Number of training steps
        resolution: Training resolution
        lora_rank: LoRA rank dimension
        learning_rate: Learning rate
        **kwargs: Additional configuration overrides
        
    Returns:
        TrainingJobConfig object
    """
    
    api = TrainingWorkerAPI()
    return api.create_simple_job(
        job_id=job_id,
        training_type="FLUX_LORA",
        dataset_path=dataset_path,
        output_path=output_path,
        max_train_steps=max_train_steps,
        resolution=resolution,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        memory_preset=gpu_memory_tier,
        **kwargs
    )


def sdxl_lora_job(
    job_id: str,
    dataset_path: str,
    output_path: str,
    max_train_steps: int = 1000,
    resolution: int = 1024,
    lora_rank: int = 8,
    learning_rate: float = 1e-4,
    batch_size: int = 2,
    **kwargs
) -> TrainingJobConfig:
    """
    Create an SDXL LoRA job with sensible defaults.
    
    Args:
        job_id: Unique job identifier
        dataset_path: Path to dataset directory
        output_path: Path for training outputs
        max_train_steps: Number of training steps
        resolution: Training resolution
        lora_rank: LoRA rank dimension
        learning_rate: Learning rate
        batch_size: Training batch size
        **kwargs: Additional configuration overrides
        
    Returns:
        TrainingJobConfig object
    """
    
    api = TrainingWorkerAPI()
    return api.create_simple_job(
        job_id=job_id,
        training_type="SDXL_LORA",
        dataset_path=dataset_path,
        output_path=output_path,
        max_train_steps=max_train_steps,
        resolution=resolution,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        batch_size=batch_size,
        **kwargs
    )


def sdxl_textual_inversion_job(
    job_id: str,
    dataset_path: str,
    output_path: str,
    placeholder_token: str,
    max_train_steps: int = 1000,
    resolution: int = 1024,
    learning_rate: float = 5e-4,
    batch_size: int = 1,
    **kwargs
) -> TrainingJobConfig:
    """
    Create an SDXL Textual Inversion job with sensible defaults.
    
    Args:
        job_id: Unique job identifier
        dataset_path: Path to dataset directory
        output_path: Path for training outputs
        placeholder_token: Token to train (e.g., "<my_concept>")
        max_train_steps: Number of training steps
        resolution: Training resolution
        learning_rate: Learning rate
        batch_size: Training batch size
        **kwargs: Additional configuration overrides
        
    Returns:
        TrainingJobConfig object
    """
    
    api = TrainingWorkerAPI()
    return api.create_simple_job(
        job_id=job_id,
        training_type="SDXL_TEXTUAL_INVERSION",
        dataset_path=dataset_path,
        output_path=output_path,
        max_train_steps=max_train_steps,
        resolution=resolution,
        learning_rate=learning_rate,
        batch_size=batch_size,
        config_overrides={
            "placeholder_token": placeholder_token,
            **kwargs
        }
    )


def sdxl_finetune_job(
    job_id: str,
    dataset_path: str,
    output_path: str,
    max_train_steps: int = 1000,
    resolution: int = 1024,
    learning_rate: float = 1e-5,
    batch_size: int = 1,
    **kwargs
) -> TrainingJobConfig:
    """
    Create an SDXL Finetune job with sensible defaults.
    
    Args:
        job_id: Unique job identifier
        dataset_path: Path to dataset directory
        output_path: Path for training outputs
        max_train_steps: Number of training steps
        resolution: Training resolution
        learning_rate: Learning rate (lower for finetune)
        batch_size: Training batch size
        **kwargs: Additional configuration overrides
        
    Returns:
        TrainingJobConfig object
    """
    
    api = TrainingWorkerAPI()
    return api.create_simple_job(
        job_id=job_id,
        training_type="SDXL_FINETUNE",
        dataset_path=dataset_path,
        output_path=output_path,
        max_train_steps=max_train_steps,
        resolution=resolution,
        learning_rate=learning_rate,
        batch_size=batch_size,
        **kwargs
    )


# Memory-optimized FLUX presets for different hardware
def flux_lora_24gb(
    job_id: str,
    dataset_path: str,
    output_path: str,
    max_train_steps: int = 1000,
    **kwargs
) -> TrainingJobConfig:
    """
    Create a FLUX LoRA job optimized for 24GB GPU (RTX 4090, A5000).
    """
    return flux_lora_job(
        job_id=job_id,
        dataset_path=dataset_path,
        output_path=output_path,
        gpu_memory_tier="24gb",
        max_train_steps=max_train_steps,
        resolution=768,  # Conservative resolution
        lora_rank=8,     # Smaller rank for faster training
        **kwargs
    )


def flux_lora_40gb(
    job_id: str,
    dataset_path: str,
    output_path: str,
    max_train_steps: int = 1000,
    **kwargs
) -> TrainingJobConfig:
    """
    Create a FLUX LoRA job optimized for 40GB GPU (A100).
    """
    return flux_lora_job(
        job_id=job_id,
        dataset_path=dataset_path,
        output_path=output_path,
        gpu_memory_tier="40gb",
        max_train_steps=max_train_steps,
        resolution=768,
        lora_rank=16,    # Balanced rank
        **kwargs
    )


def flux_lora_80gb(
    job_id: str,
    dataset_path: str,
    output_path: str,
    max_train_steps: int = 1000,
    **kwargs
) -> TrainingJobConfig:
    """
    Create a FLUX LoRA job optimized for 80GB+ GPU (H100, A100 80GB).
    """
    return flux_lora_job(
        job_id=job_id,
        dataset_path=dataset_path,
        output_path=output_path,
        gpu_memory_tier="80gb",
        max_train_steps=max_train_steps,
        resolution=1024,  # Higher resolution
        lora_rank=32,     # Higher rank for better quality
        **kwargs
    )


# Quick training functions for common use cases
def quick_flux_training(
    job_id: str,
    dataset_path: str,
    output_path: str,
    gpu_memory_gb: float,
    quality_preference: str = "balanced",
    **kwargs
) -> TrainingJobConfig:
    """
    Create a FLUX job with automatic GPU optimization.
    
    Args:
        job_id: Job identifier
        dataset_path: Dataset path
        output_path: Output path
        gpu_memory_gb: Available GPU memory in GB
        quality_preference: "fast", "balanced", or "quality"
        **kwargs: Additional overrides
    """
    
    # Determine GPU tier and settings based on memory
    if gpu_memory_gb >= 80:
        gpu_tier = "80gb"
        base_resolution = 1024
        base_rank = 32
    elif gpu_memory_gb >= 40:
        gpu_tier = "40gb"
        base_resolution = 768
        base_rank = 16
    elif gpu_memory_gb >= 24:
        gpu_tier = "24gb"
        base_resolution = 768
        base_rank = 8
    else:
        raise ValueError(f"Insufficient GPU memory for FLUX training: {gpu_memory_gb}GB (minimum 24GB required)")
    
    # Adjust based on quality preference
    if quality_preference == "fast":
        max_train_steps = 500
        lora_rank = min(base_rank, 8)
        resolution = min(base_resolution, 768)
    elif quality_preference == "quality":
        max_train_steps = 2000
        lora_rank = base_rank * 2
        resolution = base_resolution
    else:  # balanced
        max_train_steps = 1000
        lora_rank = base_rank
        resolution = base_resolution
    
    return flux_lora_job(
        job_id=job_id,
        dataset_path=dataset_path,
        output_path=output_path,
        gpu_memory_tier=gpu_tier,
        max_train_steps=max_train_steps,
        resolution=resolution,
        lora_rank=lora_rank,
        **kwargs
    )
"""
Simple wrapper for easy integration with external systems.
"""

from typing import Any, Dict, Optional
import json
from pathlib import Path

from .training_worker import TrainingWorker, TrainingJobConfig, create_job_from_dict
from .config_presets.config_presets import (
    SUPPORTED_TRAINING_TYPES,
    get_config_for_training_type,
    validate_hardware_requirements,
)
from .config_presets.commercial_config import (
    get_commercial_preset,
    estimate_training_cost,
    recommend_preset_for_budget,
)


class TrainingWorkerAPI:
    """
    High-level API wrapper for the training worker.
    
    This class provides a simplified interface for external systems to interact
    with the training worker without needing to understand the internal details.
    """
    
    def __init__(self, worker_id: str = "api_worker"):
        self.worker = TrainingWorker(worker_id=worker_id)
    
    def get_supported_training_types(self) -> list[str]:
        """Get list of supported training types."""
        return list(SUPPORTED_TRAINING_TYPES.keys())
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        return self.worker.get_hardware_info()
    
    def check_training_feasibility(
        self, 
        training_type: str,
        target_gpu_memory_gb: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Check if a training type is feasible on the current hardware.
        
        Args:
            training_type: Type of training to check
            target_gpu_memory_gb: Override GPU memory (for planning purposes)
            
        Returns:
            Dictionary with feasibility information
        """
        gpu_memory = target_gpu_memory_gb or self.worker.gpu_memory_gb
        return validate_hardware_requirements(training_type, gpu_memory)
    
    def estimate_job_cost(
        self,
        training_type: str,
        max_train_steps: int = 1000,
        resolution: int = None,
        batch_size: int = None,
        lora_rank: int = None,
        gpu_cost_per_hour: float = 2.50,
        **config_overrides
    ) -> Dict[str, Any]:
        """
        Estimate the cost of a training job.
        
        Args:
            training_type: Type of training
            max_train_steps: Number of training steps
            resolution: Training resolution
            batch_size: Training batch size
            lora_rank: LoRA rank (for LoRA training)
            gpu_cost_per_hour: Cost per GPU hour
            **config_overrides: Additional configuration parameters
            
        Returns:
            Dictionary with cost estimates
        """
        # Set defaults based on training type
        if training_type == "FLUX_LORA":
            resolution = resolution or 768
            batch_size = batch_size or 1
            lora_rank = lora_rank or 16
        else:  # SDXL types
            resolution = resolution or 1024
            batch_size = batch_size or 2
            lora_rank = lora_rank or 8
        
        config_params = {
            "max_train_steps": max_train_steps,
            "resolution": resolution,
            "batch_size": batch_size,
            "lora_rank": lora_rank,
            **config_overrides
        }
        
        config = get_config_for_training_type(training_type, **config_params)
        return estimate_training_cost(training_type, config, gpu_cost_per_hour)
    
    def get_preset_recommendations(
        self,
        training_type: str,
        budget_usd: float,
        quality_preference: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Get preset recommendations for a budget.
        
        Args:
            training_type: Type of training
            budget_usd: Maximum budget
            quality_preference: "aggressive", "balanced", or "quality"
            
        Returns:
            Dictionary with recommendations
        """
        return recommend_preset_for_budget(training_type, budget_usd, quality_preference)
    
    def create_simple_job(
        self,
        job_id: str,
        training_type: str,
        dataset_path: str,
        output_path: str,
        max_train_steps: int = 1000,
        resolution: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        lora_rank: int = None,
        commercial_preset: str = None,
        **config_overrides
    ) -> TrainingJobConfig:
        """
        Create a simple training job configuration.
        
        Args:
            job_id: Unique job identifier
            training_type: Type of training
            dataset_path: Path to dataset directory
            output_path: Path for training outputs
            max_train_steps: Number of training steps
            resolution: Training resolution
            batch_size: Training batch size
            learning_rate: Learning rate
            lora_rank: LoRA rank (for LoRA training)
            commercial_preset: Use a commercial preset
            **config_overrides: Additional configuration parameters
            
        Returns:
            TrainingJobConfig object
        """
        
        # If using a commercial preset, get the config from there
        if commercial_preset:
            preset_config = get_commercial_preset(commercial_preset)
            # Extract relevant parameters from preset
            resolution = resolution or preset_config.get("data_loader", {}).get("resolution")
            batch_size = batch_size or preset_config.get("train_batch_size")
            learning_rate = learning_rate or preset_config.get("learning_rate")
            lora_rank = lora_rank or preset_config.get("lora_rank_dim")
            
            # Add preset-specific overrides
            config_overrides.update({k: v for k, v in preset_config.items() 
                                   if k not in ["data_loader", "job_id", "seed", "base_output_dir"]})
        
        job_config = TrainingJobConfig(
            job_id=job_id,
            training_type=training_type,
            dataset_path=dataset_path,
            output_path=output_path,
            resolution=resolution,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_train_steps=max_train_steps,
            lora_rank=lora_rank,
            config_overrides=config_overrides
        )
        
        return job_config
    
    def submit_job(self, job_config: TrainingJobConfig) -> Dict[str, Any]:
        """
        Submit a training job for processing.
        
        Args:
            job_config: Training job configuration
            
        Returns:
            Dictionary with job results
        """
        return self.worker.process_job(job_config)
    
    def quick_train(
        self,
        job_id: str,
        training_type: str,
        dataset_path: str,
        output_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Quick training method that combines job creation and submission.
        
        Args:
            job_id: Unique job identifier
            training_type: Type of training
            dataset_path: Path to dataset directory
            output_path: Path for training outputs
            **kwargs: Additional job configuration parameters
            
        Returns:
            Dictionary with job results
        """
        job_config = self.create_simple_job(
            job_id=job_id,
            training_type=training_type,
            dataset_path=dataset_path,
            output_path=output_path,
            **kwargs
        )
        
        return self.submit_job(job_config)


def create_job_from_json_file(json_path: str) -> TrainingJobConfig:
    """
    Create a training job from a JSON configuration file.
    
    Args:
        json_path: Path to JSON configuration file
        
    Returns:
        TrainingJobConfig object
    """
    with open(json_path, 'r') as f:
        job_data = json.load(f)
    
    return create_job_from_dict(job_data)


def save_job_to_json_file(job_config: TrainingJobConfig, json_path: str):
    """
    Save a training job configuration to a JSON file.
    
    Args:
        job_config: Training job configuration
        json_path: Path to output JSON file
    """
    job_data = job_config.model_dump()
    
    with open(json_path, 'w') as f:
        json.dump(job_data, f, indent=2)


# Convenience functions for common use cases
def flux_lora_job(
    job_id: str,
    dataset_path: str,
    output_path: str,
    gpu_memory_tier: str = "40gb",
    cost_optimization: str = "balanced",
    max_train_steps: int = 1000,
    resolution: int = 768,
    **kwargs
) -> TrainingJobConfig:
    """
    Create a FLUX LoRA job with sensible defaults.
    
    Args:
        job_id: Unique job identifier
        dataset_path: Path to dataset directory
        output_path: Path for training outputs
        gpu_memory_tier: "24gb", "40gb", or "80gb"
        cost_optimization: "aggressive", "balanced", or "quality"
        max_train_steps: Number of training steps
        resolution: Training resolution
        **kwargs: Additional configuration overrides
        
    Returns:
        TrainingJobConfig object
    """
    preset_name = f"flux_{gpu_memory_tier}_{cost_optimization}"
    
    api = TrainingWorkerAPI()
    return api.create_simple_job(
        job_id=job_id,
        training_type="FLUX_LORA",
        dataset_path=dataset_path,
        output_path=output_path,
        max_train_steps=max_train_steps,
        resolution=resolution,
        commercial_preset=preset_name,
        **kwargs
    )


def sdxl_lora_job(
    job_id: str,
    dataset_path: str,
    output_path: str,
    cost_optimization: str = "balanced",
    max_train_steps: int = 1000,
    resolution: int = 1024,
    **kwargs
) -> TrainingJobConfig:
    """
    Create an SDXL LoRA job with sensible defaults.
    
    Args:
        job_id: Unique job identifier
        dataset_path: Path to dataset directory
        output_path: Path for training outputs
        cost_optimization: "aggressive", "balanced", or "quality"
        max_train_steps: Number of training steps
        resolution: Training resolution
        **kwargs: Additional configuration overrides
        
    Returns:
        TrainingJobConfig object
    """
    preset_name = f"sdxl_{cost_optimization}"
    
    api = TrainingWorkerAPI()
    return api.create_simple_job(
        job_id=job_id,
        training_type="SDXL_LORA",
        dataset_path=dataset_path,
        output_path=output_path,
        max_train_steps=max_train_steps,
        resolution=resolution,
        commercial_preset=preset_name,
        **kwargs
    )
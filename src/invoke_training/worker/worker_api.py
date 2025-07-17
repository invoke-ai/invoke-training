"""
High-level API wrapper for the training worker integrated with invoke_training.
"""

from typing import Any, Dict, Optional
import json
from pathlib import Path

from .training_worker import TrainingWorker, TrainingJobConfig, create_job_from_dict
from .config_presets import (
    SUPPORTED_TRAINING_TYPES,
    get_config_for_training_type,
    validate_hardware_requirements,
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
        return self._estimate_training_cost(training_type, config, gpu_cost_per_hour)
    
    def _estimate_training_cost(
        self,
        training_type: str,
        config: Dict[str, Any],
        gpu_cost_per_hour: float = 2.50,
        overhead_factor: float = 1.2
    ) -> Dict[str, Any]:
        """
        Estimate the cost of a training job.
        
        Args:
            training_type: Type of training
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
            },
            "SDXL_TEXTUAL_INVERSION": {
                "8gb": 4.0,
                "16gb": 6.0,
                "24gb": 10.0,
            },
            "SDXL_FINETUNE": {
                "24gb": 2.0,
                "40gb": 4.0,
                "80gb": 8.0,
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
            
            base_speed = speed_estimates.get(training_type, {}).get(memory_tier, 3.0)
        
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
        memory_preset: str = None,
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
            memory_preset: Memory optimization preset
            **config_overrides: Additional configuration parameters
            
        Returns:
            TrainingJobConfig object
        """
        
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
            memory_preset=memory_preset,
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
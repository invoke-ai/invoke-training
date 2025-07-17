"""
Training worker that integrates with invoke_training pipelines.
"""

import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
import yaml
from pydantic import BaseModel, ValidationError

from invoke_training.config.pipeline_config import PipelineConfig
from invoke_training.pipelines.invoke_train import train
from invoke_training.worker.config_presets import (
    SUPPORTED_TRAINING_TYPES,
    get_config_for_training_type,
    get_resource_requirements,
    validate_hardware_requirements,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TrainingJobConfig(BaseModel):
    """Pydantic model for training job configuration validation."""
    
    job_id: str
    training_type: str
    dataset_path: str
    output_path: str
    
    # Optional configuration overrides
    resolution: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    max_train_steps: Optional[int] = None
    lora_rank: Optional[int] = None
    memory_preset: Optional[str] = None
    
    # Additional config parameters
    config_overrides: Dict[str, Any] = {}


class TrainingWorker:
    """Training worker that processes jobs using invoke_training pipelines."""
    
    def __init__(self, worker_id: str = "default"):
        self.worker_id = worker_id
        self.current_job: Optional[TrainingJobConfig] = None
        self.gpu_memory_gb = self._get_gpu_memory()
        
        logger.info(f"Training worker {worker_id} initialized")
        logger.info(f"Available GPU memory: {self.gpu_memory_gb:.1f}GB")
        logger.info(f"Available system memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    
    def _get_gpu_memory(self) -> float:
        """Get available GPU memory in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
                return gpu_memory_bytes / (1024**3)
            else:
                logger.warning("CUDA not available, assuming 0GB GPU memory")
                return 0.0
        except Exception as e:
            logger.warning(f"Could not determine GPU memory: {e}")
            return 0.0
    
    def validate_job_config(self, job_config: TrainingJobConfig) -> Dict[str, Any]:
        """
        Validate job configuration and hardware requirements.
        
        Returns:
            Dictionary with validation results
        """
        try:
            # Check if training type is supported
            if job_config.training_type not in SUPPORTED_TRAINING_TYPES:
                return {
                    "valid": False,
                    "error": f"Unsupported training type: {job_config.training_type}",
                    "supported_types": list(SUPPORTED_TRAINING_TYPES.keys())
                }
            
            # Check hardware requirements
            hardware_validation = validate_hardware_requirements(
                job_config.training_type, 
                self.gpu_memory_gb
            )
            
            # Check dataset path exists
            if not Path(job_config.dataset_path).exists():
                return {
                    "valid": False,
                    "error": f"Dataset path does not exist: {job_config.dataset_path}"
                }
            
            # Create output directory if it doesn't exist
            output_path = Path(job_config.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            return {
                "valid": hardware_validation["valid"],
                "hardware_validation": hardware_validation,
                "message": "Job configuration validated successfully" if hardware_validation["valid"] else hardware_validation["message"]
            }
            
        except Exception as e:
            logger.error(f"Error validating job config: {e}")
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}"
            }
    
    def create_pipeline_config(self, job_config: TrainingJobConfig) -> PipelineConfig:
        """
        Create a PipelineConfig from job parameters using existing invoke_training structure.
        
        Args:
            job_config: Validated job configuration
            
        Returns:
            PipelineConfig object ready for training
        """
        # Generate configuration dictionary using presets
        config_params = {
            "max_train_steps": job_config.max_train_steps or 1000,
            "base_output_dir": job_config.output_path,
        }
        
        # Add optional parameters if provided
        if job_config.resolution:
            config_params["resolution"] = job_config.resolution
        if job_config.batch_size:
            config_params["batch_size"] = job_config.batch_size
        if job_config.learning_rate:
            config_params["learning_rate"] = job_config.learning_rate
        if job_config.lora_rank:
            config_params["lora_rank"] = job_config.lora_rank
        if job_config.memory_preset:
            config_params["memory_preset"] = job_config.memory_preset
        
        # Add dataset configuration
        config_params["data_loader"] = {
            "dataset": {
                "type": "IMAGE_CAPTION_JSONL_DATASET",
                "jsonl_path": str(Path(job_config.dataset_path) / "data.jsonl")
            }
        }
        
        # Apply any config overrides
        config_params.update(job_config.config_overrides)
        
        # Generate the full config using the appropriate preset
        training_config_dict = get_config_for_training_type(
            job_config.training_type,
            **config_params
        )
        
        # Set job-specific metadata
        training_config_dict["job_id"] = job_config.job_id
        training_config_dict["seed"] = hash(job_config.job_id) % 2**31  # Reproducible seed from job ID
        
        # Create PipelineConfig object using the existing validation system
        from pydantic import TypeAdapter
        pipeline_adapter: TypeAdapter[PipelineConfig] = TypeAdapter(PipelineConfig)
        pipeline_config = pipeline_adapter.validate_python(training_config_dict)
        
        return pipeline_config
    
    def process_job(self, job_config: TrainingJobConfig) -> Dict[str, Any]:
        """
        Process a training job end-to-end using invoke_training pipelines.
        
        Args:
            job_config: Training job configuration
            
        Returns:
            Dictionary with job results
        """
        try:
            self.current_job = job_config
            logger.info(f"Processing job {job_config.job_id} - Type: {job_config.training_type}")
            
            # Validate job configuration
            validation_result = self.validate_job_config(job_config)
            if not validation_result["valid"]:
                logger.error(f"Job validation failed: {validation_result}")
                return {
                    "status": "failed",
                    "error": "validation_failed",
                    "details": validation_result
                }
            
            # Log hardware validation info
            hardware_info = validation_result["hardware_validation"]
            if hardware_info.get("recommendations"):
                logger.warning(f"Hardware recommendations for {job_config.training_type}:")
                for rec in hardware_info["recommendations"]:
                    logger.warning(f"  - {rec}")
            
            # Create pipeline configuration
            pipeline_config = self.create_pipeline_config(job_config)
            
            # Save configuration for debugging
            config_path = Path(job_config.output_path) / "training_config.yaml"
            with open(config_path, 'w') as f:
                # Convert the PipelineConfig to dict for saving
                config_dict = pipeline_config.model_dump()
                yaml.dump(config_dict, f, default_flow_style=False)
            logger.info(f"Training configuration saved to: {config_path}")
            
            # Log training details based on type
            if job_config.training_type == "FLUX_LORA":
                logger.info("Starting FLUX LoRA training")
                logger.info(f"GPU Memory: {self.gpu_memory_gb:.1f}GB")
                logger.info(f"Batch size: {pipeline_config.train_batch_size}")
                logger.info(f"Gradient accumulation: {getattr(pipeline_config, 'gradient_accumulation_steps', 1)}")
                logger.info(f"Resolution: {getattr(pipeline_config.data_loader, 'resolution', 768)}")
                
                # Log FLUX-specific optimizations
                if hasattr(pipeline_config, 'cache_vae_outputs'):
                    logger.info(f"Memory optimizations:")
                    logger.info(f"  - Cache VAE outputs: {getattr(pipeline_config, 'cache_vae_outputs', False)}")
                    logger.info(f"  - Cache text encoder outputs: {getattr(pipeline_config, 'cache_text_encoder_outputs', False)}")
                    logger.info(f"  - Gradient checkpointing: {getattr(pipeline_config, 'gradient_checkpointing', False)}")
                    logger.info(f"  - Weight dtype: {getattr(pipeline_config, 'weight_dtype', 'unknown')}")
            
            # Run training using the existing invoke_training pipeline system
            train(pipeline_config)
            
            logger.info(f"Job {job_config.job_id} completed successfully")
            return {
                "status": "completed", 
                "type": job_config.training_type,
                "output_path": job_config.output_path,
                "config_path": str(config_path)
            }
            
        except Exception as e:
            logger.error(f"Job {job_config.job_id} failed: {e}")
            logger.error(traceback.format_exc())
            return {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        finally:
            self.current_job = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current worker status."""
        return {
            "worker_id": self.worker_id,
            "current_job": self.current_job.job_id if self.current_job else None,
            "gpu_memory_gb": self.gpu_memory_gb,
            "system_memory_gb": psutil.virtual_memory().total / (1024**3),
            "supported_training_types": list(SUPPORTED_TRAINING_TYPES.keys()),
            "available": self.current_job is None
        }
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get detailed hardware information."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "gpu": {
                "memory_gb": self.gpu_memory_gb,
                "cuda_available": self._check_cuda_available()
            },
            "system": {
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_percent_used": memory.percent,
                "cpu_count": psutil.cpu_count(),
                "disk_total_gb": disk.total / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
            }
        }
    
    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


def create_job_from_dict(job_data: Dict[str, Any]) -> TrainingJobConfig:
    """Create a TrainingJobConfig from a dictionary."""
    try:
        return TrainingJobConfig(**job_data)
    except ValidationError as e:
        logger.error(f"Invalid job configuration: {e}")
        raise ValueError(f"Invalid job configuration: {e}")


def main():
    """Main entry point for running the training worker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Invoke Training Worker with FLUX and SDXL support")
    parser.add_argument("--worker-id", default="default", help="Worker ID identifier")
    parser.add_argument("--job-config", help="Path to job configuration YAML file")
    parser.add_argument("--list-types", action="store_true", help="List supported training types")
    parser.add_argument("--hardware-info", action="store_true", help="Show hardware information")
    
    args = parser.parse_args()
    
    # Create worker
    worker = TrainingWorker(worker_id=args.worker_id)
    
    if args.list_types:
        print("Supported training types:")
        for training_type in SUPPORTED_TRAINING_TYPES:
            requirements = get_resource_requirements(training_type)
            print(f"  - {training_type}")
            print(f"    Min GPU Memory: {requirements['min_gpu_memory_gb']}GB")
            print(f"    Recommended GPU Memory: {requirements['recommended_gpu_memory_gb']}GB")
        return
    
    if args.hardware_info:
        import json
        hardware_info = worker.get_hardware_info()
        print(json.dumps(hardware_info, indent=2))
        return
    
    if args.job_config:
        # Load and process job from file
        config_path = Path(args.job_config)
        if not config_path.exists():
            logger.error(f"Job config file not found: {config_path}")
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            job_data = yaml.safe_load(f)
        
        job_config = create_job_from_dict(job_data)
        result = worker.process_job(job_config)
        
        print(f"Job completed with status: {result['status']}")
        if result['status'] == 'failed':
            print(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    else:
        print("Training worker ready. Use --job-config to process a job.")
        print("Use --list-types to see supported training types.")
        print("Use --hardware-info to see hardware information.")


if __name__ == "__main__":
    main()
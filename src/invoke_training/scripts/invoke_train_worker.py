#!/usr/bin/env python3
"""
Invoke Training Worker - Distributed training with FLUX and SDXL support.

This script provides worker functionality for processing training jobs using the
existing invoke_training pipeline infrastructure with enhanced job management,
hardware validation, and cost estimation capabilities.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from invoke_training.worker import TrainingWorkerAPI, flux_lora_job, sdxl_lora_job
from invoke_training.worker.training_worker import create_job_from_dict
from invoke_training.worker.config_presets import (
    SUPPORTED_TRAINING_TYPES,
    get_resource_requirements,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Invoke Training Worker with FLUX and SDXL support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check hardware and supported types
  %(prog)s --hardware-info
  %(prog)s --list-types
  
  # Process a training job from config file
  %(prog)s --job-config job.yaml
  
  # Create and run a quick FLUX job
  %(prog)s --quick-flux --job-id my_flux --dataset /path/to/data --output /path/to/output
  
  # Estimate training costs
  %(prog)s --estimate-cost --type FLUX_LORA --steps 1000
        """
    )
    
    # Worker configuration
    parser.add_argument("--worker-id", default="default", help="Worker ID identifier")
    
    # Job processing
    parser.add_argument("--job-config", type=Path, help="Path to job configuration YAML/JSON file")
    
    # Information commands
    parser.add_argument("--list-types", action="store_true", help="List supported training types")
    parser.add_argument("--hardware-info", action="store_true", help="Show hardware information")
    parser.add_argument("--check-feasibility", choices=list(SUPPORTED_TRAINING_TYPES.keys()),
                        help="Check if a training type is feasible on current hardware")
    
    # Cost estimation
    parser.add_argument("--estimate-cost", action="store_true", help="Estimate training costs")
    parser.add_argument("--type", choices=list(SUPPORTED_TRAINING_TYPES.keys()),
                        help="Training type for cost estimation")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--gpu-cost", type=float, default=2.50, help="GPU cost per hour (USD)")
    
    # Quick job creation
    parser.add_argument("--quick-flux", action="store_true", help="Create and run a quick FLUX LoRA job")
    parser.add_argument("--quick-sdxl", action="store_true", help="Create and run a quick SDXL LoRA job")
    parser.add_argument("--job-id", help="Job ID for quick jobs")
    parser.add_argument("--dataset", type=Path, help="Dataset path for quick jobs")
    parser.add_argument("--output", type=Path, help="Output path for quick jobs")
    parser.add_argument("--gpu-tier", choices=["24gb", "40gb", "80gb"], default="40gb",
                        help="GPU memory tier for FLUX jobs")
    parser.add_argument("--resolution", type=int, help="Training resolution")
    parser.add_argument("--lora-rank", type=int, help="LoRA rank dimension")
    
    return parser.parse_args()


def show_hardware_info(api: TrainingWorkerAPI):
    """Show detailed hardware information."""
    print("=== Hardware Information ===")
    hardware_info = api.get_hardware_info()
    
    print(f"GPU:")
    print(f"  Memory: {hardware_info['gpu']['memory_gb']:.1f}GB")
    print(f"  CUDA Available: {hardware_info['gpu']['cuda_available']}")
    
    print(f"System:")
    print(f"  Memory: {hardware_info['system']['memory_total_gb']:.1f}GB total, "
          f"{hardware_info['system']['memory_available_gb']:.1f}GB available")
    print(f"  CPU Cores: {hardware_info['system']['cpu_count']}")
    print(f"  Disk: {hardware_info['system']['disk_free_gb']:.1f}GB free")


def show_supported_types():
    """Show supported training types with requirements."""
    print("=== Supported Training Types ===")
    for training_type in SUPPORTED_TRAINING_TYPES:
        requirements = get_resource_requirements(training_type)
        print(f"\n{training_type}:")
        print(f"  Min GPU Memory: {requirements['min_gpu_memory_gb']}GB")
        print(f"  Recommended GPU Memory: {requirements['recommended_gpu_memory_gb']}GB")
        print(f"  Min System Memory: {requirements['min_system_memory_gb']}GB")
        print(f"  Disk Space: {requirements['disk_space_gb']}GB")
        
        if training_type == "FLUX_LORA":
            multiplier = requirements.get('training_speed_multiplier', 1.0)
            print(f"  Training Speed: ~{multiplier:.1f}x relative to SDXL")


def check_feasibility(api: TrainingWorkerAPI, training_type: str):
    """Check training feasibility for a specific type."""
    print(f"=== Feasibility Check: {training_type} ===")
    
    feasibility = api.check_training_feasibility(training_type)
    
    status_icons = {
        "insufficient": "❌",
        "minimal": "⚠️",
        "good": "✅"
    }
    
    level = feasibility.get("level", "unknown")
    icon = status_icons.get(level, "❓")
    
    print(f"{icon} Status: {feasibility['message']}")
    
    if feasibility.get("recommendations"):
        print("\nRecommendations:")
        for rec in feasibility["recommendations"]:
            print(f"  • {rec}")


def estimate_cost(api: TrainingWorkerAPI, training_type: str, steps: int, gpu_cost: float):
    """Estimate training costs."""
    print(f"=== Cost Estimation: {training_type} ===")
    
    try:
        cost_estimate = api.estimate_job_cost(
            training_type=training_type,
            max_train_steps=steps,
            gpu_cost_per_hour=gpu_cost
        )
        
        print(f"Training Steps: {steps}")
        print(f"Estimated Cost: ${cost_estimate['estimated_cost_usd']:.2f}")
        print(f"Training Time: {cost_estimate['training_hours']:.1f} hours")
        print(f"Training Speed: {cost_estimate['steps_per_minute']:.1f} steps/minute")
        print(f"Memory Tier: {cost_estimate['memory_tier']}")
        print(f"Cost per Step: ${cost_estimate['cost_per_step']:.4f}")
        
    except Exception as e:
        print(f"Error estimating cost: {e}")


def process_job_config(api: TrainingWorkerAPI, config_path: Path):
    """Process a job from configuration file."""
    if not config_path.exists():
        print(f"Error: Job config file not found: {config_path}")
        sys.exit(1)
    
    # Load configuration
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() == '.json':
            job_data = json.load(f)
        else:  # Assume YAML
            job_data = yaml.safe_load(f)
    
    print(f"=== Processing Job from {config_path} ===")
    print(f"Job ID: {job_data.get('job_id', 'unknown')}")
    print(f"Training Type: {job_data.get('training_type', 'unknown')}")
    
    try:
        job_config = create_job_from_dict(job_data)
        result = api.submit_job(job_config)
        
        print(f"\n=== Job Results ===")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'completed':
            print(f"Output Path: {result.get('output_path', 'unknown')}")
            print(f"Config Saved: {result.get('config_path', 'unknown')}")
        elif result['status'] == 'failed':
            print(f"Error: {result.get('error', 'Unknown error')}")
            if result.get('details'):
                print(f"Details: {result['details']}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error processing job: {e}")
        sys.exit(1)


def quick_flux_job(api: TrainingWorkerAPI, args):
    """Create and run a quick FLUX LoRA job."""
    if not all([args.job_id, args.dataset, args.output]):
        print("Error: --job-id, --dataset, and --output are required for quick FLUX jobs")
        sys.exit(1)
    
    print(f"=== Quick FLUX LoRA Job ===")
    print(f"Job ID: {args.job_id}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"GPU Tier: {args.gpu_tier}")
    
    kwargs = {}
    if args.resolution:
        kwargs['resolution'] = args.resolution
    if args.lora_rank:
        kwargs['lora_rank'] = args.lora_rank
    
    try:
        job_config = flux_lora_job(
            job_id=args.job_id,
            dataset_path=str(args.dataset),
            output_path=str(args.output),
            gpu_memory_tier=args.gpu_tier,
            max_train_steps=args.steps,
            **kwargs
        )
        
        result = api.submit_job(job_config)
        
        print(f"\n=== Results ===")
        print(f"Status: {result['status']}")
        if result['status'] == 'completed':
            print(f"Training completed successfully!")
        else:
            print(f"Training failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def quick_sdxl_job(api: TrainingWorkerAPI, args):
    """Create and run a quick SDXL LoRA job."""
    if not all([args.job_id, args.dataset, args.output]):
        print("Error: --job-id, --dataset, and --output are required for quick SDXL jobs")
        sys.exit(1)
    
    print(f"=== Quick SDXL LoRA Job ===")
    print(f"Job ID: {args.job_id}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    
    kwargs = {}
    if args.resolution:
        kwargs['resolution'] = args.resolution
    if args.lora_rank:
        kwargs['lora_rank'] = args.lora_rank
    
    try:
        job_config = sdxl_lora_job(
            job_id=args.job_id,
            dataset_path=str(args.dataset),
            output_path=str(args.output),
            max_train_steps=args.steps,
            **kwargs
        )
        
        result = api.submit_job(job_config)
        
        print(f"\n=== Results ===")
        print(f"Status: {result['status']}")
        if result['status'] == 'completed':
            print(f"Training completed successfully!")
        else:
            print(f"Training failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create API instance
    api = TrainingWorkerAPI(worker_id=args.worker_id)
    
    # Handle information commands
    if args.hardware_info:
        show_hardware_info(api)
        return
    
    if args.list_types:
        show_supported_types()
        return
    
    if args.check_feasibility:
        check_feasibility(api, args.check_feasibility)
        return
    
    if args.estimate_cost:
        if not args.type:
            print("Error: --type is required for cost estimation")
            sys.exit(1)
        estimate_cost(api, args.type, args.steps, args.gpu_cost)
        return
    
    # Handle job processing
    if args.job_config:
        process_job_config(api, args.job_config)
        return
    
    if args.quick_flux:
        quick_flux_job(api, args)
        return
    
    if args.quick_sdxl:
        quick_sdxl_job(api, args)
        return
    
    # No action specified, show help
    print("Invoke Training Worker ready!")
    print("\nUse one of the following options:")
    print("  --hardware-info     Show hardware information")
    print("  --list-types        List supported training types")
    print("  --job-config FILE   Process job from configuration file")
    print("  --quick-flux        Create and run a quick FLUX job")
    print("  --quick-sdxl        Create and run a quick SDXL job")
    print("\nFor more options, use --help")


if __name__ == "__main__":
    main()
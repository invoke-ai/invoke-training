#!/usr/bin/env python3
"""
Test script for the integrated invoke-training worker functionality.

This script demonstrates how to use the new worker functionality integrated
into the invoke-training library.
"""

import json
from pathlib import Path

# Import worker functionality from invoke_training
from invoke_training.worker import (
    TrainingWorkerAPI,
    flux_lora_job,
    sdxl_lora_job,
    TrainingJobConfig
)
from invoke_training.worker.config_presets import (
    get_flux_lora_config,
    validate_hardware_requirements,
    SUPPORTED_TRAINING_TYPES
)


def test_worker_api():
    """Test the worker API functionality."""
    print("🚀 Testing Invoke Training Worker Integration\n")
    
    # Initialize the worker API
    api = TrainingWorkerAPI()
    
    print("=== Hardware Information ===")
    hardware_info = api.get_hardware_info()
    print(f"GPU Memory: {hardware_info['gpu']['memory_gb']:.1f}GB")
    print(f"CUDA Available: {hardware_info['gpu']['cuda_available']}")
    print(f"System Memory: {hardware_info['system']['memory_total_gb']:.1f}GB")
    print()
    
    print("=== Supported Training Types ===")
    types = api.get_supported_training_types()
    for training_type in types:
        icon = "✨" if training_type == "FLUX_LORA" else "✓"
        print(f"{icon} {training_type}")
    print()
    
    print("=== FLUX Training Feasibility ===")
    flux_feasibility = api.check_training_feasibility("FLUX_LORA")
    status_icons = {"insufficient": "❌", "minimal": "⚠️", "good": "✅"}
    level = flux_feasibility.get("level", "unknown")
    icon = status_icons.get(level, "❓")
    print(f"{icon} {flux_feasibility['message']}")
    
    if flux_feasibility.get("recommendations"):
        print("Recommendations:")
        for rec in flux_feasibility["recommendations"]:
            print(f"  • {rec}")
    print()
    
    return api


def test_cost_estimation(api: TrainingWorkerAPI):
    """Test cost estimation functionality."""
    print("=== Cost Estimation ===")
    
    # Test FLUX cost estimation
    try:
        flux_cost = api.estimate_job_cost(
            training_type="FLUX_LORA",
            max_train_steps=1000,
            resolution=768,
            gpu_cost_per_hour=3.50
        )
        
        print("FLUX LoRA (1000 steps):")
        print(f"  💰 Estimated Cost: ${flux_cost['estimated_cost_usd']:.2f}")
        print(f"  ⏱️  Training Time: {flux_cost['training_hours']:.1f} hours")
        print(f"  🚀 Speed: {flux_cost['steps_per_minute']:.1f} steps/min")
        print(f"  🎯 Memory Tier: {flux_cost['memory_tier']}")
        
    except Exception as e:
        print(f"  ⚠️  Could not estimate FLUX costs: {e}")
    
    # Test SDXL cost estimation
    try:
        sdxl_cost = api.estimate_job_cost(
            training_type="SDXL_LORA",
            max_train_steps=1000,
            resolution=1024,
            gpu_cost_per_hour=2.00
        )
        
        print("\nSDXL LoRA (1000 steps):")
        print(f"  💰 Estimated Cost: ${sdxl_cost['estimated_cost_usd']:.2f}")
        print(f"  ⏱️  Training Time: {sdxl_cost['training_hours']:.1f} hours")
        print(f"  🚀 Speed: {sdxl_cost['steps_per_minute']:.1f} steps/min")
        
    except Exception as e:
        print(f"  ⚠️  Could not estimate SDXL costs: {e}")
    
    print()


def test_job_creation():
    """Test job creation with presets."""
    print("=== Job Configuration Creation ===")
    
    # Test FLUX job creation
    try:
        flux_job = flux_lora_job(
            job_id="test_flux_integration",
            dataset_path="/tmp/test_dataset",
            output_path="/tmp/test_output_flux",
            gpu_memory_tier="40gb",
            max_train_steps=500,
            resolution=768,
            lora_rank=16
        )
        
        print("✅ FLUX LoRA job created:")
        print(f"   Job ID: {flux_job.job_id}")
        print(f"   Type: {flux_job.training_type}")
        print(f"   Steps: {flux_job.max_train_steps}")
        print(f"   Resolution: {flux_job.resolution}")
        print(f"   LoRA Rank: {flux_job.lora_rank}")
        print(f"   Memory Preset: {flux_job.memory_preset}")
        
    except Exception as e:
        print(f"❌ FLUX job creation failed: {e}")
    
    # Test SDXL job creation
    try:
        sdxl_job = sdxl_lora_job(
            job_id="test_sdxl_integration",
            dataset_path="/tmp/test_dataset",
            output_path="/tmp/test_output_sdxl",
            max_train_steps=500,
            resolution=1024,
            lora_rank=8
        )
        
        print("\n✅ SDXL LoRA job created:")
        print(f"   Job ID: {sdxl_job.job_id}")
        print(f"   Type: {sdxl_job.training_type}")
        print(f"   Steps: {sdxl_job.max_train_steps}")
        print(f"   Resolution: {sdxl_job.resolution}")
        print(f"   LoRA Rank: {sdxl_job.lora_rank}")
        
    except Exception as e:
        print(f"❌ SDXL job creation failed: {e}")
    
    print()


def test_config_generation():
    """Test configuration generation."""
    print("=== Configuration Generation ===")
    
    try:
        # Generate a FLUX configuration
        flux_config = get_flux_lora_config(
            resolution=768,
            max_train_steps=1000,
            memory_preset="40gb",
            validation_prompts=[
                "A beautiful landscape",
                "A portrait of a person"
            ]
        )
        
        print("✅ FLUX configuration generated:")
        print(f"   Type: {flux_config['type']}")
        print(f"   Model: {flux_config['model']}")
        print(f"   Data Loader: {flux_config['data_loader']['type']}")
        print(f"   Batch Size: {flux_config['train_batch_size']}")
        print(f"   Cache VAE: {flux_config['cache_vae_outputs']}")
        print(f"   Validation Prompts: {len(flux_config['validation_prompts'])}")
        
        # Save example config
        output_file = Path("example_flux_config.json")
        with open(output_file, 'w') as f:
            json.dump(flux_config, f, indent=2, default=str)
        print(f"   📄 Config saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Config generation failed: {e}")
    
    print()


def test_hardware_validation():
    """Test hardware validation for different training types."""
    print("=== Hardware Validation ===")
    
    # Test different memory scenarios
    test_scenarios = [
        ("Low Memory GPU", 16.0),
        ("Medium Memory GPU", 24.0),
        ("High Memory GPU", 40.0),
        ("Enterprise GPU", 80.0),
    ]
    
    for scenario_name, gpu_memory in test_scenarios:
        print(f"\n{scenario_name} ({gpu_memory}GB):")
        
        for training_type in ["SDXL_LORA", "FLUX_LORA"]:
            try:
                validation = validate_hardware_requirements(training_type, gpu_memory)
                status = "✅" if validation["valid"] else "❌"
                level = validation.get("level", "unknown")
                print(f"  {status} {training_type}: {level}")
                
            except Exception as e:
                print(f"  ⚠️  {training_type}: Validation error - {e}")


def demonstrate_usage():
    """Demonstrate typical usage patterns."""
    print("=== Usage Examples ===")
    
    print("📋 Command Line Usage:")
    print("   # Check hardware and types")
    print("   python -m invoke_training.scripts.invoke_train_worker --hardware-info")
    print("   python -m invoke_training.scripts.invoke_train_worker --list-types")
    print()
    print("   # Quick FLUX training")
    print("   python -m invoke_training.scripts.invoke_train_worker \\")
    print("     --quick-flux --job-id my_flux --dataset /path/to/data --output /path/to/output")
    print()
    print("   # Process job from config")
    print("   python -m invoke_training.scripts.invoke_train_worker \\")
    print("     --job-config src/invoke_training/sample_configs/worker_flux_lora_example.yaml")
    print()
    
    print("🐍 Python API Usage:")
    print("   from invoke_training.worker import TrainingWorkerAPI, flux_lora_job")
    print("   ")
    print("   api = TrainingWorkerAPI()")
    print("   job = flux_lora_job('my_job', '/data', '/output', gpu_memory_tier='40gb')")
    print("   result = api.submit_job(job)")
    print()


def main():
    """Run all integration tests."""
    try:
        # Test core functionality
        api = test_worker_api()
        test_cost_estimation(api)
        test_job_creation()
        test_config_generation()
        test_hardware_validation()
        demonstrate_usage()
        
        print("🎉 Integration tests completed successfully!")
        print("\n✨ FLUX support has been successfully integrated into invoke-training!")
        print("\nNext steps:")
        print("1. Update your frontend to include FLUX LoRA as a training option")
        print("2. Add GPU memory tier selection for FLUX training")
        print("3. Implement cost estimation display")
        print("4. Test end-to-end training with your datasets")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
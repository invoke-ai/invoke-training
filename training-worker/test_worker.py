#!/usr/bin/env python3
"""
Simple test script for the training worker with FLUX support.
"""

import json
from pathlib import Path

from training_worker import TrainingWorkerAPI, flux_lora_job, sdxl_lora_job
from training_worker.config_presets import get_commercial_preset, estimate_training_cost


def test_hardware_info():
    """Test hardware information retrieval."""
    print("=== Hardware Information ===")
    api = TrainingWorkerAPI()
    
    hardware_info = api.get_hardware_info()
    print(f"GPU Memory: {hardware_info['gpu']['memory_gb']:.1f}GB")
    print(f"CUDA Available: {hardware_info['gpu']['cuda_available']}")
    print(f"System Memory: {hardware_info['system']['memory_total_gb']:.1f}GB")
    print(f"CPU Count: {hardware_info['system']['cpu_count']}")
    print()


def test_supported_types():
    """Test supported training types."""
    print("=== Supported Training Types ===")
    api = TrainingWorkerAPI()
    
    types = api.get_supported_training_types()
    for training_type in types:
        print(f"✓ {training_type}")
    print()


def test_flux_feasibility():
    """Test FLUX training feasibility."""
    print("=== FLUX Training Feasibility ===")
    api = TrainingWorkerAPI()
    
    feasibility = api.check_training_feasibility("FLUX_LORA")
    print(f"FLUX Feasible: {feasibility['valid']}")
    print(f"Level: {feasibility.get('level', 'unknown')}")
    print(f"Message: {feasibility['message']}")
    
    if feasibility.get('recommendations'):
        print("Recommendations:")
        for rec in feasibility['recommendations']:
            print(f"  - {rec}")
    print()


def test_cost_estimation():
    """Test cost estimation."""
    print("=== Cost Estimation ===")
    api = TrainingWorkerAPI()
    
    # Test FLUX cost estimation
    flux_cost = api.estimate_job_cost(
        training_type="FLUX_LORA",
        max_train_steps=1000,
        resolution=768,
        batch_size=1,
        lora_rank=16,
        gpu_cost_per_hour=3.50
    )
    
    print("FLUX LoRA (1000 steps):")
    print(f"  Estimated Cost: ${flux_cost['estimated_cost_usd']:.2f}")
    print(f"  Training Time: {flux_cost['training_hours']:.1f} hours")
    print(f"  Speed: {flux_cost['steps_per_minute']:.1f} steps/min")
    print(f"  Memory Tier: {flux_cost['memory_tier']}")
    
    # Test SDXL cost estimation
    sdxl_cost = api.estimate_job_cost(
        training_type="SDXL_LORA",
        max_train_steps=1000,
        resolution=1024,
        batch_size=2,
        lora_rank=8,
        gpu_cost_per_hour=2.00
    )
    
    print("\nSDXL LoRA (1000 steps):")
    print(f"  Estimated Cost: ${sdxl_cost['estimated_cost_usd']:.2f}")
    print(f"  Training Time: {sdxl_cost['training_hours']:.1f} hours")
    print(f"  Speed: {sdxl_cost['steps_per_minute']:.1f} steps/min")
    print()


def test_commercial_presets():
    """Test commercial presets."""
    print("=== Commercial Presets ===")
    
    # Test FLUX preset
    flux_config = get_commercial_preset("flux_40gb_balanced")
    print("FLUX 40GB Balanced Preset:")
    print(f"  Type: {flux_config['type']}")
    print(f"  Batch Size: {flux_config['train_batch_size']}")
    print(f"  Steps: {flux_config['max_train_steps']}")
    print(f"  LoRA Rank: {flux_config['lora_rank_dim']}")
    print(f"  Resolution: {flux_config['data_loader']['resolution']}")
    print(f"  Cache VAE: {flux_config['cache_vae_outputs']}")
    
    # Test SDXL preset
    sdxl_config = get_commercial_preset("sdxl_balanced")
    print("\nSDXL Balanced Preset:")
    print(f"  Type: {sdxl_config['type']}")
    print(f"  Batch Size: {sdxl_config['train_batch_size']}")
    print(f"  Steps: {sdxl_config['max_train_steps']}")
    print(f"  LoRA Rank: {sdxl_config['lora_rank_dim']}")
    print(f"  Resolution: {sdxl_config['data_loader']['resolution']}")
    print()


def test_budget_recommendations():
    """Test budget-based recommendations."""
    print("=== Budget Recommendations ===")
    api = TrainingWorkerAPI()
    
    recommendations = api.get_preset_recommendations(
        training_type="FLUX_LORA",
        budget_usd=10.0,
        quality_preference="balanced"
    )
    
    print(f"Recommendations for $10 budget (FLUX LoRA):")
    print(f"Total options: {recommendations['total_options']}")
    
    for i, rec in enumerate(recommendations['recommendations'], 1):
        preset_name = rec['preset_name']
        cost = rec['cost_estimate']['estimated_cost_usd']
        hours = rec['cost_estimate']['training_hours']
        steps = rec['config']['max_train_steps']
        
        print(f"  {i}. {preset_name}")
        print(f"     Cost: ${cost:.2f} ({hours:.1f} hours)")
        print(f"     Steps: {steps}")
    print()


def test_job_creation():
    """Test job creation."""
    print("=== Job Creation ===")
    
    # Create FLUX job
    flux_job = flux_lora_job(
        job_id="test_flux_001",
        dataset_path="/tmp/test_dataset",
        output_path="/tmp/test_output_flux",
        gpu_memory_tier="40gb",
        cost_optimization="balanced",
        max_train_steps=100,  # Small for testing
        resolution=768
    )
    
    print("FLUX LoRA Job Created:")
    print(f"  Job ID: {flux_job.job_id}")
    print(f"  Type: {flux_job.training_type}")
    print(f"  Steps: {flux_job.max_train_steps}")
    print(f"  Resolution: {flux_job.resolution}")
    
    # Create SDXL job
    sdxl_job = sdxl_lora_job(
        job_id="test_sdxl_001",
        dataset_path="/tmp/test_dataset",
        output_path="/tmp/test_output_sdxl",
        cost_optimization="balanced",
        max_train_steps=100,
        resolution=1024
    )
    
    print("\nSDXL LoRA Job Created:")
    print(f"  Job ID: {sdxl_job.job_id}")
    print(f"  Type: {sdxl_job.training_type}")
    print(f"  Steps: {sdxl_job.max_train_steps}")
    print(f"  Resolution: {sdxl_job.resolution}")
    print()


def test_config_export():
    """Test exporting job configurations."""
    print("=== Configuration Export ===")
    
    api = TrainingWorkerAPI()
    
    # Create a job config
    job_config = api.create_simple_job(
        job_id="export_test",
        training_type="FLUX_LORA",
        dataset_path="/tmp/dataset",
        output_path="/tmp/output",
        resolution=768,
        max_train_steps=500,
        commercial_preset="flux_40gb_balanced"
    )
    
    # Get the full training config
    training_config = api.worker.create_training_config(job_config)
    
    # Save to file for inspection
    output_file = Path("test_flux_config.json")
    with open(output_file, 'w') as f:
        json.dump(training_config, f, indent=2, default=str)
    
    print(f"Full FLUX training config exported to: {output_file}")
    print(f"Config size: {len(training_config)} parameters")
    print(f"Model: {training_config.get('model', 'unknown')}")
    print(f"Data loader type: {training_config.get('data_loader', {}).get('type', 'unknown')}")
    print()


def main():
    """Run all tests."""
    print("🚀 Testing Training Worker with FLUX Support\n")
    
    try:
        test_hardware_info()
        test_supported_types()
        test_flux_feasibility()
        test_cost_estimation()
        test_commercial_presets()
        test_budget_recommendations()
        test_job_creation()
        test_config_export()
        
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
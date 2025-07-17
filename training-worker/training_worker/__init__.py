"""
Training Worker with FLUX and SDXL Support

A distributed AI model training worker that supports FLUX and SDXL model fine-tuning
with LoRA, Textual Inversion, and full fine-tuning capabilities.
"""

__version__ = "0.2.0"

from .training_worker import TrainingWorker, TrainingJobConfig
from .wrapper import TrainingWorkerAPI, flux_lora_job, sdxl_lora_job

__all__ = [
    "TrainingWorker",
    "TrainingJobConfig", 
    "TrainingWorkerAPI",
    "flux_lora_job",
    "sdxl_lora_job",
]
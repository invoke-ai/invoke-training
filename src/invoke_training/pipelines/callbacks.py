from abc import ABC
from enum import Enum


class ModelType(Enum):
    # At first glance, it feels like these model types should be further broken down into separate enums (e.g.
    # base_model, model_type, checkpoint_format). But, I haven't yet come up with a taxonomy that feels sufficiently
    # future-proof. So, for now, there is one enum for each file type that invoke-training can produce.

    # A Flux LoRA model in PEFT format.
    FLUX_LORA_PEFT = "FLUX_LORA_PEFT"
    # A Flux LoRA model in Kohya format.
    FLUX_LORA_KOHYA = "FLUX_LORA_KOHYA"

    # A Stable Diffusion 1.x LoRA model in Kohya format.
    SD1_LORA_KOHYA = "SD1_LORA_KOHYA"
    # A Stable Diffusion 1.x LoRA model in PEFT format.
    SD1_LORA_PEFT = "SD1_LORA_PEFT"
    # A Stable Diffusion XL LoRA model in Kohya format.
    SDXL_LORA_KOHYA = "SDXL_LORA_KOHYA"
    # A Stable Diffusion XL LoRA model in PEFT format.
    SDXL_LORA_PEFT = "SDXL_LORA_PEFT"

    # A Stable Diffusion 1.x Textual Inversion model.
    SD1_TEXTUAL_INVERSION = "SD1_TEXTUAL_INVERSION"
    # A Stable Diffusion XL Textual Inversion model.
    SDXL_TEXTUAL_INVERSION = "SDXL_TEXTUAL_INVERSION"

    # A Stable Diffusion 1.x UNet checkpoint in diffusers format.
    SD1_UNET_DIFFUSERS = "SD1_UNET_DIFFUSERS"
    # A Stable Diffusion XL UNet checkpoint in diffusers format.
    SDXL_UNET_DIFFUSERS = "SDXL_UNET_DIFFUSERS"
    # A full Stable Diffusion XL checkpoint in diffusers format.
    SDXL_FULL_DIFFUSERS = "SDXL_FULL_DIFFUSERS"


class ModelCheckpoint:
    """A single model checkpoint."""

    def __init__(self, file_path: str, model_type: ModelType):
        self.file_path = file_path
        self.model_type = model_type


class TrainingCheckpoint:
    """A training checkpoint. May contain multiple model checkpoints if multiple models are being trained
    simultaneously.
    """

    def __init__(self, models: list[ModelCheckpoint], epoch: int, step: int):
        self.models = models
        self.epoch = epoch
        self.step = step


class ValidationImage:
    def __init__(self, file_path: str, prompt: str, image_idx: int):
        """A single validation image.

        Args:
            file_path (str): Path to the image file.
            prompt (str): The prompt used to generate the image.
            image_idx (int): The index of this image in the current validation set (i.e. in the set of images generated
                with the same prompt at the same validation point).
        """
        self.file_path = file_path
        self.prompt = prompt
        self.image_idx = image_idx


class ValidationImages:
    def __init__(self, images: list[ValidationImage], epoch: int, step: int):
        """A collection of validation images.

        Args:
            images (list[ValidationImage]): The validation images.
            epoch (int): The last completed epoch at the time that these images were generated.
            step (int): The last completed training step at the time that these images were generated.
        """
        self.images = images
        self.epoch = epoch
        self.step = step


class PipelineCallbacks(ABC):
    def on_save_checkpoint(self, checkpoint: TrainingCheckpoint):
        pass

    def on_save_validation_images(self, images: ValidationImages):
        pass

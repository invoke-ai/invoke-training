from typing import Annotated, Literal, Optional, Union

from pydantic import Field

from invoke_training.config.pipelines.base_pipeline_config import BasePipelineConfig
from invoke_training.config.shared.data.data_loader_config import (
    DreamboothSDDataLoaderConfig,
    ImageCaptionSDDataLoaderConfig,
)
from invoke_training.config.shared.optimizer.optimizer_config import OptimizerConfig


class LoRATrainingConfig(BasePipelineConfig):
    """The base configuration for any LoRA training run."""

    model: str = "runwayml/stable-diffusion-v1-5"
    """Name or path of the base model to train. Can be in diffusers format, or a single stable diffusion checkpoint
    file. (E.g. 'runwayml/stable-diffusion-v1-5', '/path/to/realisticVisionV51_v51VAE.safetensors', etc. )
    """

    train_unet: bool = True
    """Whether to add LoRA layers to the UNet model and train it.
    """

    train_text_encoder: bool = True
    """Whether to add LoRA layers to the text encoder and train it.
    """

    text_encoder_learning_rate: Optional[float] = None
    """The learning rate to use for the text encoder model. If set, this overrides the optimizer's default learning
    rate.
    """

    unet_learning_rate: Optional[float] = None
    """The learning rate to use for the UNet model. If set, this overrides the optimizer's default learning rate.
    """

    train_unet_non_attention_blocks: bool = False
    """Whether to inject LoRA layers into the non-attention UNet blocks for training. Enabling will produce a more
    expressive LoRA model at the cost of slower training, higher training VRAM requirements, and a larger LoRA weight
    file.
    """

    lora_rank_dim: int = 4
    """The rank dimension to use for the LoRA layers. Increasing the rank dimension increases the model's expressivity,
    but also increases the size of the generated LoRA model.
    """

    cache_text_encoder_outputs: bool = False
    """If True, the text encoder(s) will be applied to all of the captions in the dataset before starting training and
    the results will be cached to disk. This reduces the VRAM requirements during training (don't have to keep the
    text encoders in VRAM), and speeds up training  (don't have to run the text encoders for each training example).
    This option can only be enabled if `train_text_encoder == False` and there are no caption augmentations being
    applied.
    """

    cache_vae_outputs: bool = False
    """If True, the VAE will be applied to all of the images in the dataset before starting training and the results
    will be cached to disk. This reduces the VRAM requirements during training (don't have to keep the VAE in VRAM), and
    speeds up training (don't have to run the VAE encoding step). This option can only be enabled if all
    non-deterministic image augmentations are disabled (i.e. center_crop=True, random_flip=False).
    """

    enable_cpu_offload_during_validation: bool = False
    """If True, models will be kept in CPU memory and loaded into GPU memory one-by-one while generating validation
    images. This reduces VRAM requirements at the cost of slower generation of validation images.
    """

    gradient_accumulation_steps: int = 1
    """The number of gradient steps to accumulate before each weight update. This value is passed to Hugging Face
    Accelerate. This is an alternative to increasing the batch size when training with limited VRAM.
    """

    mixed_precision: Optional[Literal["no", "fp16", "bf16", "fp8"]] = None
    """The mixed precision mode to use ('no','fp16','bf16 or 'fp8'). This value is passed to Hugging Face Accelerate.
    See accelerate.Accelerator for more details.
    """

    xformers: bool = False
    """If true, use xformers for more efficient attention blocks.
    """

    gradient_checkpointing: bool = False
    """Whether or not to use gradient checkpointing to save memory at the expense of a slower backward pass. Enabling
    gradient checkpointing slows down training by ~20%.
    """

    max_train_steps: int = 5000
    """Total number of training steps to perform. One training step is one gradient update.
    """

    save_every_n_epochs: Optional[int] = 1
    """The interval (in epochs) at which to save checkpoints. If None, checkpoint won't be triggered by this setting. It
    is recommend to only set one of save_every_n_epochs and save_every_n_steps to a non-None value.
    """

    save_every_n_steps: Optional[int] = None
    """The interval (in steps) at which to save checkpoints. If None, checkpoint won't be triggered by this setting. It
    is recommend to only set one of save_every_n_epochs and save_every_n_steps to a non-None value.
    """

    max_checkpoints: Optional[int] = None
    """The maximum number of checkpoints to keep. New checkpoints will replace earlier checkpoints to stay under this
    limit. Note that this limit is applied to 'step' and 'epoch' checkpoints separately.
    """

    prediction_type: Optional[Literal["epsilon", "v_prediction"]] = None
    """The prediction_type that will be used for training. Choose between 'epsilon' or 'v_prediction' or leave 'None'.
    If 'None', the prediction type of the scheduler: `noise_scheduler.config.prediction_type` is used.
    """

    max_grad_norm: Optional[float] = None
    """Max gradient norm for clipping. Set to None for no clipping.
    """

    validation_prompts: list[str] = []
    """A list of prompts that will be used to generate images throughout training for the purpose of tracking progress.
    See also 'validate_every_n_epochs'.
    """

    num_validation_images_per_prompt: int = 4
    """The number of validation images to generate for each prompt in 'validation_prompts'. Careful, validation can
    become quite slow if this number is too large.
    """

    validate_every_n_epochs: int = 1
    """The interval (in epochs) at which validation images will be generated.
    """

    train_batch_size: int = 4
    """The training batch size.
    """


class FinetuneLoRASDConfig(LoRATrainingConfig):
    type: Literal["FINETUNE_LORA_SD"] = "FINETUNE_LORA_SD"
    optimizer: OptimizerConfig
    data_loader: Annotated[
        Union[ImageCaptionSDDataLoaderConfig, DreamboothSDDataLoaderConfig], Field(discriminator="type")
    ]


class FinetuneLoRASDXLConfig(LoRATrainingConfig):
    type: Literal["FINETUNE_LORA_SDXL"] = "FINETUNE_LORA_SDXL"
    optimizer: OptimizerConfig
    data_loader: Annotated[
        Union[ImageCaptionSDDataLoaderConfig, DreamboothSDDataLoaderConfig], Field(discriminator="type")
    ]

    vae_model: Optional[str] = None
    """The name of the Hugging Face Hub VAE model to train against. This will override the VAE bundled with the base
    model (specified by the `model` parameter). This config option is provided for SDXL models, because SDXL shipped
    with a VAE that produces NaNs in fp16 mode, so it is common to replace this VAE with a fixed version.
    """

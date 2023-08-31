import typing

from pydantic import BaseModel

from invoke_training.training.config.data_config import (
    DreamBoothDataLoaderConfig,
    ImageCaptionDataLoaderConfig,
)
from invoke_training.training.config.optimizer_config import OptimizerConfig


class TrainingOutputConfig(BaseModel):
    """Configuration for a training run's output."""

    # The output directory where the training outputs (model checkpoints, logs,
    # intermediate predictions) will be written. A subdirectory will be created
    # with a timestamp for each new training run.
    base_output_dir: str

    # The integration to report results and logs to ('all', 'tensorboard',
    # 'wandb', or 'comet_ml'). This value is passed to Hugging Face Accelerate.
    # See accelerate.Accelerator.log_with for more details.
    report_to: typing.Optional[typing.Literal["all", "tensorboard", "wandb", "comet_ml"]] = "tensorboard"

    # The file type to save the model as.
    # Note that "ckpt" and "pt" are alternative file extensions for the same
    # file format.
    save_model_as: typing.Literal["ckpt", "pt", "safetensors"] = "safetensors"


class LoRATrainingConfig(BaseModel):
    """The base configuration for any LoRA training run."""

    # The name of the Hugging Face Hub model to train against.
    model: str = "runwayml/stable-diffusion-v1-5"

    # A seed for reproducible training.
    seed: typing.Optional[int] = None

    # Whether to add LoRA layers to the UNet model and train it.
    train_unet: bool = True

    # Whether to add LoRA layers to the text encoder and train it.
    train_text_encoder: bool = True

    # Whether to inject LoRA layers into the non-attention UNet blocks for training. Enabling will produce a more
    # expressive LoRA model at the cost of slower training, higher training VRAM requirements, and a larger LoRA weight
    # file.
    train_unet_non_attention_blocks: bool = False

    # The rank dimension to use for the LoRA layers. Increasing the rank dimension increases the model's expressivity,
    # but also increases the size of the generated LoRA model.
    lora_rank_dim: int = 4

    # If True, the text encoder(s) will be applied to all of the captions in the dataset before starting training and
    # the results will be cached to disk. This reduces the VRAM requirements during training (don't have to keep the
    # text encoders in VRAM), and speeds up training  (don't have to run the text encoders for each training example).
    # This option can only be enabled if `train_text_encoder == False` and there are no caption augmentations being
    # applied.
    cache_text_encoder_outputs: bool = False

    # If True, the VAE will be applied to all of the images in the dataset before starting training and the results will
    # be cached to disk. This reduces the VRAM requirements during training (don't have to keep the VAE in VRAM), and
    # speeds up training (don't have to run the VAE encoding step). This option can only be enabled if all
    # non-deterministic image augmentations are disabled (i.e. center_crop=True, random_flip=False).
    cache_vae_outputs: bool = False

    # If True, models will be kept in CPU memory and loaded into GPU memory one-by-one while generating validation
    # images. This reduces VRAM requirements at the cost of slower generation of validation images.
    enable_cpu_offload_during_validation: bool = False

    # The number of gradient steps to accumulate before each weight update. This value is passed to Hugging Face
    # Accelerate. This is an alternative to increasing the batch size when training with limited VRAM.
    gradient_accumulation_steps: int = 1

    # The mixed precision mode to use ('no','fp16','bf16 or 'fp8'). This value is passed to Hugging Face Accelerate. See
    # accelerate.Accelerator for more details.
    mixed_precision: typing.Optional[typing.Literal["no", "fp16", "bf16", "fp8"]] = None

    # If true, use xformers for more efficient attention blocks.
    xformers: bool = False

    # Whether or not to use gradient checkpointing to save memory at the expense of a slower backward pass. Enabling
    # gradient checkpointing slows down training by ~20%.
    gradient_checkpointing: bool = False

    # Total number of training steps to perform. (One training step is one gradient update.)
    max_train_steps: int = 5000

    # The interval (in epochs) at which to save checkpoints. If None, checkpoint won't be triggered by this setting. It
    # is recommend to only set one of save_every_n_epochs and save_every_n_steps to a non-None value.
    save_every_n_epochs: typing.Optional[int] = 1

    # The interval (in steps) at which to save checkpoints. If None, checkpoint won't be triggered by this setting. It
    # is recommend to only set one of save_every_n_epochs and save_every_n_steps to a non-None value.
    save_every_n_steps: typing.Optional[int] = None

    # The maximum number of checkpoints to keep. New checkpoints will replace earlier checkpoints to stay under this
    # limit. Note that this limit is applied to 'step' and 'epoch' checkpoints separately.
    max_checkpoints: typing.Optional[int] = None

    # The prediction_type that will be used for training. Choose between 'epsilon' or 'v_prediction' or leave 'None'.
    # If 'None', the prediction type of the scheduler: `noise_scheduler.config.prediction_type` is used.
    prediction_type: typing.Optional[typing.Literal["epsilon", "v_prediction"]] = None

    # Max gradient norm for clipping. Set to None for no clipping.
    max_grad_norm: typing.Optional[float] = 1.0

    # A list of prompts that will be used to generate images throughout training for the purpose of tracking progress.
    # See also 'validate_every_n_epochs'.
    validation_prompts: list[str] = []

    # The number of validation images to generate for each prompt in 'validation_prompts'. Careful, validation can
    # become quite slow if this number is too large.
    num_validation_images_per_prompt: int = 4

    # The interval (in epochs) at which validation images will be generated.
    validate_every_n_epochs: int = 1

    # The training batch size.
    train_batch_size: int = 4


class FinetuneLoRAConfig(LoRATrainingConfig):
    output: TrainingOutputConfig
    optimizer: OptimizerConfig
    dataset: ImageCaptionDataLoaderConfig


class FinetuneLoRASDXLConfig(FinetuneLoRAConfig):
    # The name of the Hugging Face Hub VAE model to train against. This will override the VAE bundled with the base
    # model (specified by the `model` parameter). This config option is provided for SDXL models, because SDXL shipped
    # with a VAE that produces NaNs in fp16 mode, so it is common to replace this VAE with a fixed version.
    vae_model: typing.Optional[str] = None


class DreamBoothLoRAConfig(LoRATrainingConfig):
    output: TrainingOutputConfig
    optimizer: OptimizerConfig
    dataset: DreamBoothDataLoaderConfig


class DreamBoothLoRASDXLConfig(DreamBoothLoRAConfig):
    # The name of the Hugging Face Hub VAE model to train against. This will override the VAE bundled with the base
    # model (specified by the `model` parameter). This config option is provided for SDXL models, because SDXL shipped
    # with a VAE that produces NaNs in fp16 mode, so it is common to replace this VAE with a fixed version.
    vae_model: typing.Optional[str] = None

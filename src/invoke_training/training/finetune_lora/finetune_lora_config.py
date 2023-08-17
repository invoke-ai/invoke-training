import typing

from pydantic import BaseModel


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


class TrainingOptimizerConfig(BaseModel):
    """Configuration for the training optimizer. (Currently, only torch.optim.AdamW is supported.)"""

    # Initial learning rate to use (after the potential warmup period).
    learning_rate: float = 1e-4

    # Adam optimizer params.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8

    # The number of warmup steps in the learning rate scheduler. Only applied to
    # schedulers that support warmup. See lr_scheduler.
    lr_warmup_steps: int = 0

    # The learning rate scheduler to use.
    lr_scheduler: typing.Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ] = "constant"


class DatasetConfig(BaseModel):
    # The name of a Hugging Face dataset.
    # One of dataset_name and dataset_dir should be set (dataset_name takes precedence).
    # See also: dataset_config_name.
    dataset_name: typing.Optional[str] = None

    # The directory to load a dataset from. The dataset is expected to be in
    # Hugging Face imagefolder format (https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder).
    # One of 'dataset_name' and 'dataset_dir' should be set ('dataset_name' takes precedence).
    dataset_dir: typing.Optional[str] = None

    # The Hugging Face dataset config name. Leave as None if there's only one config.
    # This parameter is only used if dataset_name is set.
    dataset_config_name: typing.Optional[str] = None

    # The Hugging Face cache directory to use for dataset downloads.
    # If None, the default value will be used (usually '~/.cache/huggingface/datasets').
    hf_cache_dir: typing.Optional[str] = None

    # The name of the dataset column that contains image paths.
    image_column: str = "image"

    # The name of the dataset column that contains captions.
    caption_column: str = "text"

    # The resolution for input images. All of the images in the dataset will be resized to this (square) resolution.
    resolution: int = 512

    # If True, input images will be center-cropped to resolution.
    # If False, input images will be randomly cropped to resolution.
    center_crop: bool = False

    # Whether random flip augmentations should be applied to input images.
    random_flip: bool = False

    # Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    dataloader_num_workers: int = 0


class FinetuneLoRAConfig(BaseModel):
    """The configuration for a LoRA training run."""

    output: TrainingOutputConfig

    optimizer: TrainingOptimizerConfig

    dataset: DatasetConfig

    ##################
    # General Configs
    ##################

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

    # Whether or not to use gradient checkpointing to save memory at the expense of a slower backward pass.
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


class FinetuneLoRASDXLConfig(FinetuneLoRAConfig):
    # The name of the Hugging Face Hub VAE model to train against. This will override the VAE bundled with the base
    # model (specified by the `model` parameter). This config option is provided for SDXL models, because SDXL shipped
    # with a VAE that produces NaNs in fp16 mode, so it is common to replace this VAE with a fixed version.
    vae_model: typing.Optional[str] = None

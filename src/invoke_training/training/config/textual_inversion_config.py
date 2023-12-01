import typing

from pydantic import BaseModel

from invoke_training.training.config.data_config import ImageCaptionDataLoaderConfig
from invoke_training.training.config.optimizer_config import OptimizerConfig
from invoke_training.training.config.training_output_config import TrainingOutputConfig


class TextualInversionTrainingConfig(BaseModel):
    """The base configuration for any Textual Inversion training run."""

    # Name or path of the base model to train. Can be in diffusers format, or a single stable diffusion checkpoint file.
    # (E.g. 'runwayml/stable-diffusion-v1-5', 'stabilityai/stable-diffusion-xl-base-1.0',
    # '/path/to/realisticVisionV51_v51VAE.safetensors', etc. )
    model: str = "runwayml/stable-diffusion-v1-5"

    # A seed for reproducible training.
    seed: typing.Optional[int] = None

    # The number of textual inversion placeholder vectors that will be used to learn the concept.
    num_vectors: int = 1

    # The special word to associate the learned embeddings with. You must use this trigger word in your prompt at
    # inference time.
    placeholder_token: str

    # A vocabulary token to use as an initializer for the placeholder token(s). It should be a single word that roughly
    # describes the object or style that you're trying to train on. Must map to a single tokenizer token.
    initializer_token: str

    # Whether you're training the model to learn a new "style" or a new "object".
    learnable_property: typing.Literal["object", "style"] = "object"

    # The learning rate.
    learning_rate: float = 0.00001

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


class TextualInversionConfig(TextualInversionTrainingConfig):
    output: TrainingOutputConfig
    optimizer: OptimizerConfig
    dataset: ImageCaptionDataLoaderConfig

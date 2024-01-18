from typing import Literal, Optional

from invoke_training.config.pipelines.base_pipeline_config import BasePipelineConfig
from invoke_training.config.shared.data.data_loader_config import TextualInversionSDDataLoaderConfig
from invoke_training.config.shared.optimizer.optimizer_config import OptimizerConfig


class TextualInversionTrainingConfig(BasePipelineConfig):
    """The base configuration for any Textual Inversion training run."""

    model: str
    """Name or path of the base model to train. Can be in diffusers format, or a single stable diffusion checkpoint
    file. (E.g. `"runwayml/stable-diffusion-v1-5"`, `"stabilityai/stable-diffusion-xl-base-1.0"`,
    `"/path/to/local/model.safetensors"`, etc.)

    The model architecture must match the training pipeline being run. For example, if running a
    Textual Inversion SDXL pipeline, then `model` must refer to an SDXL model.
    """

    hf_variant: str | None = "fp16"
    """The Hugging Face Hub model variant to use. Only applies if `model` is a Hugging Face Hub model name.
    """

    # Helpful discussion for understanding how this works at inference time:
    # https://github.com/huggingface/diffusers/pull/3144#discussion_r1172413509
    num_vectors: int = 1
    """Note: `num_vectors` can be overridden by `initial_phrase`.

    The number of textual inversion embedding vectors that will be used to learn the concept.

    Increasing the `num_vectors` enables the model to learn more complex concepts, but has the following drawbacks:

    - greater risk of overfitting
    - increased size of the resulting output file
    - consumes more of the prompt capacity at inference time

    Typical values for `num_vectors` are in the range [1, 16].

    As a rule of thumb, `num_vectors` can be increased as the size of the dataset increases (without overfitting).
    """

    placeholder_token: str
    """The special word to associate the learned embeddings with. Choose a unique token that is unlikely to already
    exist in the tokenizer's vocabulary.
    """

    initializer_token: Optional[str] = None
    """Note: Exactly one of `initializer_token`, `initial_embedding_file`, or `initial_phrase` should be set.

    A vocabulary token to use as an initializer for the placeholder token. It should be a single word that roughly
    describes the object or style that you're trying to train on. Must map to a single tokenizer token.

    For example, if you are training on a dataset of images of your pet dog, a good choice would be `dog`.
    """

    initial_embedding_file: Optional[str] = None
    """Note: Exactly one of `initializer_token`, `initial_embedding_file`, or `initial_phrase` should be set.

    Path to an existing TI embedding that will be used to initialize the embedding being trained. The placeholder
    token in the file must match the `placeholder_token` field.

    Either `initializer_token` or `initial_embedding_file` should be set.
    """

    initial_phrase: Optional[str] = None
    """Note: Exactly one of `initializer_token`, `initial_embedding_file`, or `initial_phrase` should be set.

    A phrase that will be used to initialize the placeholder token embedding. The phrase will be tokenized, and the
    corresponding embeddings will be used to initialize the placeholder tokens. The number of embedding vectors will be
    inferred from the length of the tokenized phrase, so keep the phrase short. The consequences of training a large
    number of embedding vectors are discussed in the `num_vectors` field documentation.

    For example, if you are training on a dataset of images of pokemon, you might use `pokemon sketch white background`.
    """

    cache_vae_outputs: bool = False
    """If True, the VAE will be applied to all of the images in the dataset before starting training and the results
    will be cached to disk. This reduces the VRAM requirements during training (don't have to keep the VAE in VRAM), and
    speeds up training (don't have to run the VAE encoding step).

    This option can only be enabled if all non-deterministic image augmentations are disabled (i.e. `center_crop=True`,
    `random_flip=False`, etc.).
    """

    enable_cpu_offload_during_validation: bool = False
    """If True, models will be kept in CPU memory and loaded into GPU memory one-by-one while generating validation
    images. This reduces VRAM requirements at the cost of slower generation of validation images.
    """

    gradient_accumulation_steps: int = 1
    """The number of gradient steps to accumulate before each weight update. This is an alternative to increasing the
    `train_batch_size` when training with limited VRAM.
    """

    mixed_precision: Optional[Literal["no", "fp16", "bf16", "fp8"]] = None
    """The mixed precision mode to use. This value is passed to Hugging Face Accelerate.
    See
    [`accelerate.Accelerator.mixed_precision`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.mixed_precision)
    for more details.
    """

    xformers: bool = False
    """If `True`, use xformers for more efficient attention blocks.
    """

    gradient_checkpointing: bool = False
    """Whether or not to use gradient checkpointing to save memory at the expense of a slower backward pass. Enabling
    gradient checkpointing slows down training by ~20%.
    """

    max_train_steps: int = 5000
    """Total number of training steps to perform. (One training step is one gradient update.)
    """

    save_every_n_epochs: Optional[int] = 1
    """The interval (in epochs) at which to save checkpoints. If `None`, checkpoint won't be triggered by this setting.
    It is recommend to only set one of `save_every_n_epochs` and `save_every_n_steps` to a non-`None` value.
    """

    save_every_n_steps: Optional[int] = None
    """The interval (in steps) at which to save checkpoints. If `None`, checkpoint won't be triggered by this setting.
    It is recommend to only set one of `save_every_n_epochs` and `save_every_n_steps` to a non-`None` value.
    """

    max_checkpoints: Optional[int] = None
    """The maximum number of checkpoints to keep. New checkpoints will replace earlier checkpoints to stay under this
    limit. Note that this limit is applied to 'step' and 'epoch' checkpoints separately.
    """

    prediction_type: Optional[Literal["epsilon", "v_prediction"]] = None
    """The prediction type that will be used for training. If `None`, the prediction type will be inferred from the
    scheduler.
    """

    max_grad_norm: Optional[float] = None
    """Maximum gradient norm for gradient clipping. Set to `None` for no clipping.
    """

    validation_prompts: list[str] = []
    """A list of prompts that will be used to generate images throughout training for the purpose of tracking progress.
    See also `validate_every_n_epochs`.
    """

    num_validation_images_per_prompt: int = 4
    """The number of validation images to generate for each prompt in `validation_prompts`. Careful, validation can
    become very slow if this number is too large.
    """

    validate_every_n_epochs: int = 1
    """The interval (in epochs) at which validation images will be generated.
    """

    train_batch_size: int = 4
    """The training batch size.
    """


class TextualInversionSDConfig(TextualInversionTrainingConfig):
    type: Literal["TEXTUAL_INVERSION_SD"] = "TEXTUAL_INVERSION_SD"
    """Must be `TEXTUAL_INVERSION_SD`. This is what differentiates training pipeline types.
    """

    optimizer: OptimizerConfig
    """Configuration for the training optimizer (algorithm, learning rate, etc.).

    See [`OptimizerConfig`][invoke_training.config.shared.optimizer.optimizer_config.OptimizerConfig] for details.
    """

    data_loader: TextualInversionSDDataLoaderConfig
    """The data configuration.

    See
    [`TextualInversionSDDataLoaderConfig`][invoke_training.config.shared.data.data_loader_config.TextualInversionSDDataLoaderConfig]
    for details.
    """


class TextualInversionSDXLConfig(TextualInversionTrainingConfig):
    type: Literal["TEXTUAL_INVERSION_SDXL"] = "TEXTUAL_INVERSION_SDXL"
    """Must be `TEXTUAL_INVERSION_SDXL`. This is what differentiates training pipeline types.
    """

    optimizer: OptimizerConfig
    """Configuration for the training optimizer (algorithm, learning rate, etc.).

    See [`OptimizerConfig`][invoke_training.config.shared.optimizer.optimizer_config.OptimizerConfig] for details.
    """

    data_loader: TextualInversionSDDataLoaderConfig
    """The data configuration.

    See
    [`TextualInversionSDDataLoaderConfig`][invoke_training.config.shared.data.data_loader_config.TextualInversionSDDataLoaderConfig]
    for details.
    """

    vae_model: Optional[str] = None
    """The name of the Hugging Face Hub VAE model to train against. If set, this will override the VAE bundled with the
    base model (specified by the `model` parameter). This config option is provided for SDXL models, because SDXL 1.0
    shipped with a VAE that produces NaNs in fp16 mode, so it is common to replace this VAE with a fixed version.
    """

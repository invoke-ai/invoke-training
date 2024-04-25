from typing import Literal

from pydantic import model_validator

from invoke_training.config.base_pipeline_config import BasePipelineConfig
from invoke_training.config.data.data_loader_config import TextualInversionSDDataLoaderConfig
from invoke_training.config.optimizer.optimizer_config import AdamOptimizerConfig, ProdigyOptimizerConfig


class SdxlTextualInversionConfig(BasePipelineConfig):
    type: Literal["SDXL_TEXTUAL_INVERSION"] = "SDXL_TEXTUAL_INVERSION"
    """Must be `SDXL_TEXTUAL_INVERSION`. This is what differentiates training pipeline types.
    """

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

    initializer_token: str | None = None
    """Note: Exactly one of `initializer_token`, `initial_embedding_file`, or `initial_phrase` should be set.

    A vocabulary token to use as an initializer for the placeholder token. It should be a single word that roughly
    describes the object or style that you're trying to train on. Must map to a single tokenizer token.

    For example, if you are training on a dataset of images of your pet dog, a good choice would be `dog`.
    """

    initial_embedding_file: str | None = None
    """Note: Exactly one of `initializer_token`, `initial_embedding_file`, or `initial_phrase` should be set.

    Path to an existing TI embedding that will be used to initialize the embedding being trained. The placeholder
    token in the file must match the `placeholder_token` field.

    Either `initializer_token` or `initial_embedding_file` should be set.
    """

    initial_phrase: str | None = None
    """Note: Exactly one of `initializer_token`, `initial_embedding_file`, or `initial_phrase` should be set.

    A phrase that will be used to initialize the placeholder token embedding. The phrase will be tokenized, and the
    corresponding embeddings will be used to initialize the placeholder tokens. The number of embedding vectors will be
    inferred from the length of the tokenized phrase, so keep the phrase short. The consequences of training a large
    number of embedding vectors are discussed in the `num_vectors` field documentation.

    For example, if you are training on a dataset of images of pokemon, you might use `pokemon sketch white background`.
    """

    optimizer: AdamOptimizerConfig | ProdigyOptimizerConfig = AdamOptimizerConfig()

    lr_scheduler: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = "constant"

    lr_warmup_steps: int = 0
    """The number of warmup steps in the learning rate scheduler. Only applied to schedulers that support warmup.
    See lr_scheduler.
    """

    min_snr_gamma: float | None = 5.0
    """Min-SNR weighting for diffusion training was introduced in https://arxiv.org/abs/2303.09556. This strategy
    improves the speed of training convergence by adjusting the weight of each sample.

    `min_snr_gamma` acts like an an upper bound on the weight of samples with low noise levels.

    If `None`, then Min-SNR weighting will not be applied. If enabled, the recommended value is `min_snr_gamma = 5.0`.
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

    mixed_precision: Literal["no", "fp16", "bf16", "fp8"] = "no"
    """The mixed precision mode to use.

    If mixed precision is enabled, then all non-trainable parameters will be cast to the specified precision. The
    trainable parameters are always kept in float32 precision to avoid issues with numerical stability.

    Recommendations:

    - `"no"`: Use this mode if you have plenty of VRAM available.
    - `"bf16"`: Use this mode if you have limited VRAM and a GPU that supports bfloat16.
    - `"fp16"`: Use this mode if you have limited VRAM and a GPU that does not support bfloat16.
    - `"fp8"`: You are likely to run into numerical stability issues with this mode. Only use this mode if you know what you are doing and are willing to work through some issues.

    This value is passed to Hugging Face Accelerate. See
    [`accelerate.Accelerator.mixed_precision`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.mixed_precision)
    for more details.
    """  # noqa: E501

    xformers: bool = False
    """If `True`, use xformers for more efficient attention blocks.
    """

    gradient_checkpointing: bool = False
    """Whether or not to use gradient checkpointing to save memory at the expense of a slower backward pass. Enabling
    gradient checkpointing slows down training by ~20%.
    """

    max_checkpoints: int | None = None
    """The maximum number of checkpoints to keep. New checkpoints will replace earlier checkpoints to stay under this
    limit. Note that this limit is applied to 'step' and 'epoch' checkpoints separately.
    """

    prediction_type: Literal["epsilon", "v_prediction"] | None = None
    """The prediction type that will be used for training. If `None`, the prediction type will be inferred from the
    scheduler.
    """

    max_grad_norm: float | None = None
    """Maximum gradient norm for gradient clipping. Set to `None` for no clipping.
    """

    validation_prompts: list[str] = []
    """A list of prompts that will be used to generate images throughout training for the purpose of tracking progress.
    """

    negative_validation_prompts: list[str] | None = None
    """A list of negative prompts that will be applied when generating validation images. If set, this list should have
    the same length as 'validation_prompts'.
    """

    num_validation_images_per_prompt: int = 4
    """The number of validation images to generate for each prompt in `validation_prompts`. Careful, validation can
    become very slow if this number is too large.
    """

    train_batch_size: int = 4
    """The training batch size.
    """

    data_loader: TextualInversionSDDataLoaderConfig
    """The data configuration.

    See
    [`TextualInversionSDDataLoaderConfig`][invoke_training.config.data.data_loader_config.TextualInversionSDDataLoaderConfig]
    for details.
    """

    vae_model: str | None = None
    """The name of the Hugging Face Hub VAE model to train against. If set, this will override the VAE bundled with the
    base model (specified by the `model` parameter). This config option is provided for SDXL models, because SDXL 1.0
    shipped with a VAE that produces NaNs in fp16 mode, so it is common to replace this VAE with a fixed version.
    """

    @model_validator(mode="after")
    def check_validation_prompts(self):
        if self.negative_validation_prompts is not None and len(self.negative_validation_prompts) != len(
            self.validation_prompts
        ):
            raise ValueError(
                f"The number of validation_prompts ({len(self.validation_prompts)}) must match the number of "
                f"negative_validation_prompts ({len(self.negative_validation_prompts)})."
            )
        return self

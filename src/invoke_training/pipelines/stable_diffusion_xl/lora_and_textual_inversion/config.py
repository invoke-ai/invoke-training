from typing import Literal

from pydantic import model_validator

from invoke_training.config.base_pipeline_config import BasePipelineConfig
from invoke_training.config.data.data_loader_config import TextualInversionSDDataLoaderConfig
from invoke_training.config.optimizer.optimizer_config import AdamOptimizerConfig, ProdigyOptimizerConfig


class SdxlLoraAndTextualInversionConfig(BasePipelineConfig):
    type: Literal["SDXL_LORA_AND_TEXTUAL_INVERSION"] = "SDXL_LORA_AND_TEXTUAL_INVERSION"

    model: str = "runwayml/stable-diffusion-v1-5"
    """Name or path of the base model to train. Can be in diffusers format, or a single stable diffusion checkpoint
    file. (E.g. 'runwayml/stable-diffusion-v1-5', '/path/to/realisticVisionV51_v51VAE.safetensors', etc. )
    """

    hf_variant: str | None = "fp16"
    """The Hugging Face Hub model variant to use. Only applies if `model` is a Hugging Face Hub model name.
    """

    lora_checkpoint_format: Literal["invoke_peft", "kohya"] = "kohya"
    """The format of the LoRA checkpoint to save. Choose between `invoke_peft` or `kohya`."""

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
    """A vocabulary token to use as an initializer for the placeholder token. It should be a single word that roughly
    describes the object or style that you're trying to train on. Must map to a single tokenizer token.

    For example, if you are training on a dataset of images of your pet dog, a good choice would be `dog`.
    """

    initial_phrase: str | None = None
    """Note: Exactly one of `initializer_token` or `initial_phrase` should be set.

    A phrase that will be used to initialize the placeholder token embedding. The phrase will be tokenized, and the
    corresponding embeddings will be used to initialize the placeholder tokens. The number of embedding vectors will be
    inferred from the length of the tokenized phrase, so keep the phrase short. The consequences of training a large
    number of embedding vectors are discussed in the `num_vectors` field documentation.

    For example, if you are training on a dataset of images of pokemon, you might use `pokemon sketch white background`.
    """

    train_unet: bool = True
    """Whether to add LoRA layers to the UNet model and train it.
    """

    train_text_encoder: bool = True
    """Whether to add LoRA layers to the text encoder and train it.
    """

    train_ti: bool = True
    """Whether to train the textual inversion embeddings."""

    ti_train_steps_ratio: float | None = None
    """The fraction of the total training steps for which the TI embeddings will be trained. For example, if we are
    training for a total of 5000 steps and `ti_train_steps_ratio=0.5`, then the TI embeddings will be trained for 2500
    steps and the will be frozen for the remaining steps.

    If `None`, then the TI embeddings will be trained for the entire duration of training.
    """

    optimizer: AdamOptimizerConfig | ProdigyOptimizerConfig = AdamOptimizerConfig()

    text_encoder_learning_rate: float = 1e-5
    """The learning rate to use for the text encoder model.
    """

    unet_learning_rate: float = 1e-4
    """The learning rate to use for the UNet model.
    """

    textual_inversion_learning_rate: float = 1e-3
    """The learning rate to use for textual inversion training of the embeddings.
    """

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

    mixed_precision: Literal["no", "fp16", "bf16", "fp8"] = "no"
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

    max_checkpoints: int | None = None
    """The maximum number of checkpoints to keep. New checkpoints will replace earlier checkpoints to stay under this
    limit. Note that this limit is applied to 'step' and 'epoch' checkpoints separately.
    """

    prediction_type: Literal["epsilon", "v_prediction"] | None = None
    """The prediction_type that will be used for training. Choose between 'epsilon' or 'v_prediction' or leave 'None'.
    If 'None', the prediction type of the scheduler: `noise_scheduler.config.prediction_type` is used.
    """

    max_grad_norm: float | None = None
    """Max gradient norm for clipping. Set to None for no clipping.
    """

    validation_prompts: list[str] = []
    """A list of prompts that will be used to generate images throughout training for the purpose of tracking progress.
    """

    negative_validation_prompts: list[str] | None = None
    """A list of negative prompts that will be applied when generating validation images. If set, this list should have
    the same length as 'validation_prompts'.
    """

    num_validation_images_per_prompt: int = 4
    """The number of validation images to generate for each prompt in 'validation_prompts'. Careful, validation can
    become quite slow if this number is too large.
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
    """The name of the Hugging Face Hub VAE model to train against. This will override the VAE bundled with the base
    model (specified by the `model` parameter).

    This config option is provided for SDXL model training, because SDXL shipped with a VAE that produces NaNs in fp16
    mode. There are two ways to work around this problem:
    - Set 'vae_dtype' to 'float32' to use the VAE in float32 mode.
    - Set 'vae_model' to a fixed VAE model that does not produce NaNs in fp16 mode.
    """

    vae_dtype: Literal["float16", "float32"] | None = None
    """The dtype to use for the VAE model. If set, then this value will override the 'mixed_precision' setting.

    This config option is provided for SDXL model training, because SDXL shipped with a VAE that produces NaNs in fp16
    mode. There are two ways to work around this problem:
    - Set 'vae_dtype' to 'float32' to use the VAE in float32 mode.
    - Set 'vae_model' to a fixed VAE model that does not produce NaNs in fp16 mode.
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

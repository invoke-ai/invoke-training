from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from invoke_training.config.base_pipeline_config import BasePipelineConfig
from invoke_training.config.data.data_loader_config import DreamboothSDDataLoaderConfig, ImageCaptionSDDataLoaderConfig
from invoke_training.config.optimizer.optimizer_config import AdamOptimizerConfig, ProdigyOptimizerConfig


class SdxlLoraConfig(BasePipelineConfig):
    type: Literal["SDXL_LORA"] = "SDXL_LORA"

    model: str = "runwayml/stable-diffusion-v1-5"
    """Name or path of the base model to train. Can be in diffusers format, or a single stable diffusion checkpoint
    file. (E.g. 'runwayml/stable-diffusion-v1-5', '/path/to/realisticVisionV51_v51VAE.safetensors', etc. )
    """

    hf_variant: str | None = "fp16"
    """The Hugging Face Hub model variant to use. Only applies if `model` is a Hugging Face Hub model name.
    """

    # Note: Pydantic handles mutable default values well:
    # https://docs.pydantic.dev/latest/concepts/models/#fields-with-non-hashable-default-values
    base_embeddings: dict[str, str] = {}
    """A mapping of embedding tokens to trained embedding file paths. These embeddings will be applied to the base model
    before training.

    Example:
    ```
    base_embeddings = {
        "bruce_the_gnome": "/path/to/bruce_the_gnome.safetensors",
    }
    ```

    Consider also adding the embedding tokens to the `data_loader.caption_prefix` if they are not already present in the
    dataset captions.

    Note that the embeddings themselves are not fine-tuned further, but they will impact the LoRA model training if they
    are referenced in the dataset captions. The list of embeddings provided here should be the same list used at
    generation time with the resultant LoRA model.
    """

    lora_checkpoint_format: Literal["invoke_peft", "kohya"] = "kohya"
    """The format of the LoRA checkpoint to save. Choose between `invoke_peft` or `kohya`."""

    train_unet: bool = True
    """Whether to add LoRA layers to the UNet model and train it.
    """

    train_text_encoder: bool = True
    """Whether to add LoRA layers to the text encoder and train it.
    """

    optimizer: AdamOptimizerConfig | ProdigyOptimizerConfig = AdamOptimizerConfig()

    text_encoder_learning_rate: float | None = None
    """The learning rate to use for the text encoder model. If set, this overrides the optimizer's default learning
    rate.
    """

    unet_learning_rate: float | None = None
    """The learning rate to use for the UNet model. If set, this overrides the optimizer's default learning rate.
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

    weight_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    """All weights (trainable and fixed) will be cast to this precision. Lower precision dtypes require less VRAM, and
    result in faster training, but are more prone to issues with numerical stability.

    Recommendations:

    - `"float32"`: Use this mode if you have plenty of VRAM available.
    - `"bfloat16"`: Use this mode if you have limited VRAM and a GPU that supports bfloat16.
    - `"float16"`: Use this mode if you have limited VRAM and a GPU that does not support bfloat16.

    See also [`mixed_precision`][invoke_training.pipelines.stable_diffusion_xl.lora.config.SdxlLoraConfig.mixed_precision].
    """  # noqa: E501

    mixed_precision: Literal["no", "fp16", "bf16", "fp8"] = "no"
    """The mixed precision mode to use.

    If mixed precision is enabled, then all non-trainable parameters will be cast to the specified `weight_dtype`, and
    trainable parameters are kept in float32 precision to avoid issues with numerical stability.

    This value is passed to Hugging Face Accelerate. See
    [`accelerate.Accelerator.mixed_precision`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.mixed_precision)
    for more details.
    """  # noqa: E501

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
    See also 'validate_every_n_epochs'.
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

    data_loader: Annotated[
        Union[ImageCaptionSDDataLoaderConfig, DreamboothSDDataLoaderConfig], Field(discriminator="type")
    ]

    vae_model: str | None = None
    """The name of the Hugging Face Hub VAE model to train against. This will override the VAE bundled with the base
    model (specified by the `model` parameter). This config option is provided for SDXL models, because SDXL shipped
    with a VAE that produces NaNs in fp16 mode, so it is common to replace this VAE with a fixed version.
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

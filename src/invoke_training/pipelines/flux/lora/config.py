from typing import Annotated, Literal, Union
from pydantic import Field

from invoke_training._shared.flux.lora_checkpoint_utils import (
    FLUX_TRANSFORMER_TARGET_MODULES,
    TEXT_ENCODER_TARGET_MODULES,
)
from invoke_training.config.base_pipeline_config import BasePipelineConfig
from invoke_training.config.data.data_loader_config import ImageCaptionFluxDataLoaderConfig
from invoke_training.config.optimizer.optimizer_config import (
    AdamOptimizerConfig,
    ProdigyOptimizerConfig,
)

class FluxLoraConfig(BasePipelineConfig):
    type: Literal["FLUX_LORA"] = "FLUX_LORA"

    model: str = "black-forest-labs/FLUX.1-dev"
    """Name or path of the base model to train. Can be in diffusers format, or a single Flux.1-dev checkpoint
    file. (E.g. 'black-forest-labs/FLUX.1-dev', '/path/to/flux.1-dev.safetensors', etc. )
    """

    transformer_path: str | None = None
    """Path to the custom transformer .safetensors file. If not provided, the default black-forest-labs/FLUX.1-dev
    transformer will be used.
    """

    text_encoder_1_path: str | None = None
    """Path to the custom CLIP text encoder .safetensors file. If not provided, the default openai/clip-vit-base-patch32
    text encoder will be used.
    """

    text_encoder_2_path: str | None = None
    """Path to the custom T5 text encoder .safetensors file. If not provided, the default google/t5-v1_1-xl text encoder
     will be used.
     """

    lora_checkpoint_format: Literal["invoke_peft", "kohya"] = "kohya"
    """The format of the LoRA checkpoint to save. Choose between `invoke_peft` or `kohya`."""

    train_transformer: bool = True
    """Whether to add LoRA layers to the FluxTransformer2DModel and train it.
    """

    train_text_encoder: bool = False
    """Whether to add LoRA layers to the text encoder and train it.
    """

    optimizer: AdamOptimizerConfig | ProdigyOptimizerConfig = AdamOptimizerConfig()

    text_encoder_learning_rate: float | None = 1e-4
    """The learning rate to use for the text encoder model. If set, this overrides the optimizer's default learning
    rate.
    """

    transformer_learning_rate: float | None = 1e-4
    """The learning rate to use for the transformer model. If set, this overrides the optimizer's default learning
    rate.
    """

    lr_scheduler: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = "constant_with_warmup"

    lr_warmup_steps: int = 10
    """The number of warmup steps in the learning rate scheduler. Only applied to schedulers that support warmup.
    See lr_scheduler.
    """

    min_snr_gamma: float | None = None
    """Min-SNR weighting for diffusion training was introduced in https://arxiv.org/abs/2303.09556. This strategy
    improves the speed of training convergence by adjusting the weight of each sample.

    `min_snr_gamma` acts like an an upper bound on the weight of samples with low noise levels.

    If `None`, then Min-SNR weighting will not be applied. If enabled, the recommended value is `min_snr_gamma = 5.0`.
    """

    lora_rank_dim: int = 4
    """The rank dimension to use for the LoRA layers. Increasing the rank dimension increases the model's expressivity,
    but also increases the size of the generated LoRA model.
    """

    flux_lora_target_modules: list[str] = FLUX_TRANSFORMER_TARGET_MODULES
    """The list of target modules to apply LoRA layers to in the FluxTransformer2DModel. The default list will produce a
    highly expressive LoRA model.

    For a smaller and less expressive LoRA model, the following list is recommended:
    ```python
    flux_lora_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    ```

    The list of target modules is passed to Hugging Face's PEFT library. See
    [the docs](https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig.target_modules) for
    details.
    """

    text_encoder_lora_target_modules: list[str] = TEXT_ENCODER_TARGET_MODULES
    """The list of target modules to apply LoRA layers to in the CLIP text encoder. The default list will produce a
    highly expressive LoRA model.

    For a smaller and less expressive LoRA model, the following list is recommended:
    ```python
    text_encoder_lora_target_modules = ["fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj"]
    ```

    The list of target modules is passed to Hugging Face's PEFT library. See
    [the docs](https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraConfig.target_modules) for
    details.
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

    weight_dtype: Literal["float32", "float16", "bfloat16"] = "float16"
    """All weights (trainable and fixed) will be cast to this precision. Lower precision dtypes require less VRAM, and
    result in faster training, but are more prone to issues with numerical stability.

    Recommendations:

    - `"float32"`: Use this mode if you have plenty of VRAM available.
    - `"bfloat16"`: Use this mode if you have limited VRAM and a GPU that supports bfloat16.
    - `"float16"`: Use this mode if you have limited VRAM and a GPU that does not support bfloat16.

    See also [`mixed_precision`][invoke_training.pipelines.flux.lora.config.FluxLoraConfig.mixed_precision].
    """  # noqa: E501

    mixed_precision: Literal["no", "fp16", "bf16", "fp8"] = "no"
    """The mixed precision mode to use.

    If mixed precision is enabled, then all non-trainable parameters will be cast to the specified `weight_dtype`, and
    trainable parameters are kept in float32 precision to avoid issues with numerical stability.

    This value is passed to Hugging Face Accelerate. See
    [`accelerate.Accelerator.mixed_precision`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.mixed_precision)
    for more details.
    """  # noqa: E501

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

    num_validation_images_per_prompt: int = 4
    """The number of validation images to generate for each prompt in 'validation_prompts'. Careful, validation can
    become quite slow if this number is too large.
    """

    train_batch_size: int = 1
    """The training batch size.
    """

    use_masks: bool = False
    """If True, image masks will be applied to weight the loss during training. The dataset must contain masks for this
    feature to be used.
    """

    data_loader: Annotated[
        Union[ImageCaptionFluxDataLoaderConfig], Field(discriminator="type")
    ]

    timestep_sampler: Literal["shift", "uniform"] = "shift"
    """The timestep sampler to use. Choose between 'shift' or 'uniform'."""

    discrete_flow_shift: float = 3.0
    """The shift parameter for the discrete flow. Only used if `timestep_sampler == "shift"`.
    """

    sigmoid_scale: float = 1.0
    """The scale parameter for the sigmoid function. Only used if `timestep_sampler == "shift"`.
    """

    lora_scale: float | None = 1.0
    """The scale parameter for the LoRA layers. If set, this overrides the optimizer's default learning rate.
    """

    guidance_scale: float = 1.0
    """The guidance scale for the Flux model.
    """

    train_transformer: bool = True
    """Whether to train the Flux transformer (FluxTransformer2DModel) model.
    """

    clip_tokenizer_max_length: int = 77
    """The maximum length of the CLIP tokenizer. The maximum length of the CLIP tokenizer is 77.
    """

    t5_tokenizer_max_length: int = 512
    """The maximum length of the T5 tokenizer. The maximum length of the T5 tokenizer is 512.
    """

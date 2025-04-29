import os
from pathlib import Path

import peft
import torch
from diffusers import FluxTransformer2DModel
from transformers import CLIPTextModel

from invoke_training._shared.checkpoints.serialization import save_state_dict
from invoke_training._shared.checkpoints.lora_checkpoint_utils import save_multi_model_peft_checkpoint, load_multi_model_peft_checkpoint, _convert_peft_models_to_kohya_state_dict, _convert_peft_state_dict_to_kohya_state_dict


FLUX_TRANSFORMER_TARGET_MODULES = [
# double blocks
"attn.add_k_proj",
"attn.add_q_proj",
"attn.add_v_proj",
"attn.to_add_out",
"attn.to_k",
"attn.to_q",
"attn.to_v",
"attn.to_out.0",
"ff.net.0.proj",
"ff.net.2.0",
"ff_context.net.0.proj",
"ff_context.net.2.0",
# single blocks
"attn.to_k",
"attn.to_q",
"attn.to_v",

"proj_mlp",
"proj_out",
"proj_in",
]

TEXT_ENCODER_TARGET_MODULES = ["fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj"]

# Module lists copied from diffusers training script.
# These module lists will produce lighter, less expressive, LoRA models than the non-light versions.
FLUX_TRANSFORMER_TARGET_MODULES_LIGHT = ["to_k", "to_q", "to_v", "to_out.0"]
FLUX_TEXT_ENCODER_TARGET_MODULES_LIGHT = ["q_proj", "k_proj", "v_proj", "out_proj"]

FLUX_PEFT_TRANSFORMER_KEY = "transformer"
FLUX_PEFT_TEXT_ENCODER_1_KEY = "text_encoder_1"
FLUX_PEFT_TEXT_ENCODER_2_KEY = "text_encoder_2"

FLUX_KOHYA_TRANSFORMER_KEY = "lora_unet"
FLUX_KOHYA_TEXT_ENCODER_1_KEY = "lora_clip"
FLUX_KOHYA_TEXT_ENCODER_2_KEY = "lora_t5"

FLUX_PEFT_TO_KOHYA_KEYS = {
    FLUX_PEFT_TRANSFORMER_KEY: FLUX_KOHYA_TRANSFORMER_KEY,
    FLUX_PEFT_TEXT_ENCODER_1_KEY: FLUX_KOHYA_TEXT_ENCODER_1_KEY,
    FLUX_PEFT_TEXT_ENCODER_2_KEY: FLUX_KOHYA_TEXT_ENCODER_2_KEY,
}



def save_flux_peft_checkpoint(
    checkpoint_dir: Path | str,
    transformer: peft.PeftModel | None,
    text_encoder_1: peft.PeftModel | None,
    text_encoder_2: peft.PeftModel | None,
):
    models = {}
    if transformer is not None:
        models[FLUX_PEFT_TRANSFORMER_KEY] = transformer
    if text_encoder_1 is not None:
        models[FLUX_PEFT_TEXT_ENCODER_1_KEY] = text_encoder_1
    if text_encoder_2 is not None:
        models[FLUX_PEFT_TEXT_ENCODER_2_KEY] = text_encoder_2

    save_multi_model_peft_checkpoint(checkpoint_dir=checkpoint_dir, models=models)


def load_flux_peft_checkpoint(
    checkpoint_dir: Path | str, transformer: FluxTransformer2DModel, text_encoder_1: CLIPTextModel, text_encoder_2: CLIPTextModel, is_trainable: bool = False
):
    models = load_multi_model_peft_checkpoint(
        checkpoint_dir=checkpoint_dir,
        models={
            FLUX_PEFT_TRANSFORMER_KEY: transformer,
            FLUX_PEFT_TEXT_ENCODER_1_KEY: text_encoder_1,
            FLUX_PEFT_TEXT_ENCODER_2_KEY: text_encoder_2,
        },
        is_trainable=is_trainable,
        raise_if_subdir_missing=False,
    )

    return models[FLUX_PEFT_TRANSFORMER_KEY], models[FLUX_PEFT_TEXT_ENCODER_1_KEY], models[
        FLUX_PEFT_TEXT_ENCODER_2_KEY
    ]


def save_flux_kohya_checkpoint(checkpoint_path: Path, transformer: peft.PeftModel | None, text_encoder_1: peft.PeftModel | None, text_encoder_2: peft.PeftModel | None):
    kohya_prefixes = []
    models = []
    for kohya_prefix, peft_model in zip([FLUX_KOHYA_TRANSFORMER_KEY, FLUX_KOHYA_TEXT_ENCODER_1_KEY], [transformer, text_encoder_1]):
        if peft_model is not None:
            kohya_prefixes.append(kohya_prefix)
            models.append(peft_model)

    kohya_state_dict = _convert_peft_models_to_kohya_state_dict(kohya_prefixes=kohya_prefixes, models=models)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_state_dict(kohya_state_dict, checkpoint_path)


def convert_flux_peft_checkpoint_to_kohya_state_dict(
    in_checkpoint_dir: Path,
    out_checkpoint_file: Path,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Convert Flux PEFT models to a Kohya-format LoRA state dict."""
    # Get the immediate subdirectories of the checkpoint directory. We assume that each subdirectory is a PEFT model.
    peft_model_dirs = os.listdir(in_checkpoint_dir)
    peft_model_dirs = [in_checkpoint_dir / d for d in peft_model_dirs]  # Convert to Path objects.
    peft_model_dirs = [d for d in peft_model_dirs if d.is_dir()]  # Filter out non-directories.

    if len(peft_model_dirs) == 0:
        raise ValueError(f"No checkpoint files found in directory '{in_checkpoint_dir}'.")

    kohya_state_dict = {}
    for peft_model_dir in peft_model_dirs:
        if peft_model_dir.name in FLUX_PEFT_TO_KOHYA_KEYS:
            kohya_prefix = FLUX_PEFT_TO_KOHYA_KEYS[peft_model_dir.name]
        else:
            raise ValueError(f"Unrecognized checkpoint directory: '{peft_model_dir}'.")

        # Note: This logic to load the LoraConfig and weights directly is based on how it is done here:
        # https://github.com/huggingface/peft/blob/8665e2b5719faa4e4b91749ddec09442927b53e0/src/peft/peft_model.py#L672-L689
        # This may need to be updated in the future to support other adapter types (LoKr, LoHa, etc.).
        # Also, I could see this interface breaking in the future.
        lora_config = peft.LoraConfig.from_pretrained(peft_model_dir)
        lora_weights = peft.utils.load_peft_weights(peft_model_dir, device="cpu")

        kohya_state_dict.update(
            _convert_peft_state_dict_to_kohya_state_dict(
                lora_config=lora_config, peft_state_dict=lora_weights, prefix=kohya_prefix, dtype=dtype
            )
        )

    save_state_dict(kohya_state_dict, out_checkpoint_file)


def _convert_peft_models_to_kohya_state_dict(
    kohya_prefixes: list[str], models: list[peft.PeftModel]
) -> dict[str, torch.Tensor]:
    kohya_state_dict = {}
    default_adapter_name = "default"

    for kohya_prefix, peft_model in zip(kohya_prefixes, models, strict=True):
        lora_config = peft_model.peft_config[default_adapter_name]
        assert isinstance(lora_config, peft.LoraConfig)

        state_dict = peft.get_peft_model_state_dict(peft_model, adapter_name=default_adapter_name)

        if kohya_prefix == FLUX_KOHYA_TRANSFORMER_KEY:
            state_dict = convert_diffusers_to_flux_transformer_checkpoint(state_dict)

        kohya_state_dict.update(
            _convert_peft_state_dict_to_kohya_state_dict(
                lora_config=lora_config,
                peft_state_dict=state_dict,
                prefix=kohya_prefix,
                dtype=torch.float32,
            )
        )

    return kohya_state_dict


def find_matching_key_prefix(state_dict, key_pattern):
    """
    Find if any key in the state dictionary matches the given pattern.

    Args:
        state_dict: The state dictionary to search in
        key_pattern: The pattern to look for in keys

    Returns:
        The matching prefix if found, False otherwise
    """
    base_prefix = key_pattern.split(".lora_A")[0].split(".lora_B")[0].split(".weight")[0]

    for key in state_dict.keys():
        if base_prefix in key:
            return base_prefix
    return False


def convert_layer_weights(target_dict, source_dict, source_pattern, target_pattern):
    """
    Convert weights from source pattern to target pattern if they exist.

    Args:
        target_dict: Dictionary to store converted weights
        source_dict: Source dictionary containing weights
        source_pattern: Original key pattern to search for
        target_pattern: New key pattern to use


    Returns:
        Tuple of (updated target_dict, updated source_dict)
    """
    if (original_key := find_matching_key_prefix(source_dict, source_pattern)) != False:
        # Find all keys matching the pattern
        keys_to_convert = [k for k in source_dict.keys() if original_key in k]

        for key in keys_to_convert:
            # Create replacement key
            new_key = key.replace(
                original_key, target_pattern.replace(".weight", "")
            )
            # Transfer and remove from original
            target_dict[new_key] = source_dict.pop(key)

    return target_dict, source_dict


def convert_double_transformer_block(target_dict, source_dict, prefix="", block_idx=0):
    """
    Convert weights for a double transformer block.

    Args:
        target_dict: Dictionary to store converted weights
        source_dict: Source dictionary containing weights
        prefix: Prefix for the keys in the state dictionary
        block_idx: Block index

    Returns:
        Tuple of (updated target_dict, updated source_dict)
    """
    block_prefix = f"transformer_blocks.{block_idx}."

    # Convert norms
    target_dict, source_dict = convert_layer_weights(
        target_dict,
        source_dict,
        f"{prefix}{block_prefix}norm1.linear.weight",
        f"double_blocks.{block_idx}.img_mod.lin.weight",
    )

    target_dict, source_dict = convert_layer_weights(
        target_dict,
        source_dict,
        f"{prefix}{block_prefix}norm1_context.linear.weight",
        f"double_blocks.{block_idx}.txt_mod.lin.weight",
    )

    # Convert attention weights by concatenating Q, K, V
    try:
        # Sample attention weights
        sample_q_A = source_dict.pop(f"{prefix}{block_prefix}attn.to_q.lora_A.weight")
        sample_q_B = source_dict.pop(f"{prefix}{block_prefix}attn.to_q.lora_B.weight")
        sample_k_A = source_dict.pop(f"{prefix}{block_prefix}attn.to_k.lora_A.weight")
        sample_k_B = source_dict.pop(f"{prefix}{block_prefix}attn.to_k.lora_B.weight")
        sample_v_A = source_dict.pop(f"{prefix}{block_prefix}attn.to_v.lora_A.weight")
        sample_v_B = source_dict.pop(f"{prefix}{block_prefix}attn.to_v.lora_B.weight")

        # Context attention weights
        context_q_A = source_dict.pop(f"{prefix}{block_prefix}attn.add_q_proj.lora_A.weight")
        context_q_B = source_dict.pop(f"{prefix}{block_prefix}attn.add_q_proj.lora_B.weight")
        context_k_A = source_dict.pop(f"{prefix}{block_prefix}attn.add_k_proj.lora_A.weight")
        context_k_B = source_dict.pop(f"{prefix}{block_prefix}attn.add_k_proj.lora_B.weight")
        context_v_A = source_dict.pop(f"{prefix}{block_prefix}attn.add_v_proj.lora_A.weight")
        context_v_B = source_dict.pop(f"{prefix}{block_prefix}attn.add_v_proj.lora_B.weight")

        # Concatenate Q, K, V for image and text
        target_dict[f"double_blocks.{block_idx}.img_attn.qkv.lora_A.weight"] = torch.cat(
            [sample_q_A, sample_k_A, sample_v_A], dim=0
        )
        target_dict[f"double_blocks.{block_idx}.img_attn.qkv.lora_B.weight"] = torch.cat(
            [sample_q_B, sample_k_B, sample_v_B], dim=0
        )
        target_dict[f"double_blocks.{block_idx}.txt_attn.qkv.lora_A.weight"] = torch.cat(
            [context_q_A, context_k_A, context_v_A], dim=0
        )
        target_dict[f"double_blocks.{block_idx}.txt_attn.qkv.lora_B.weight"] = torch.cat(
            [context_q_B, context_k_B, context_v_B], dim=0
        )
    except KeyError as e:
        print(f"Error processing attention weights for block {block_idx}: {e}")
        raise
    
    # Convert QK norms
    norm_keys = [
        (f"{prefix}{block_prefix}attn.norm_q.weight", f"double_blocks.{block_idx}.img_attn.norm.query_norm.scale"),
        (f"{prefix}{block_prefix}attn.norm_k.weight", f"double_blocks.{block_idx}.img_attn.norm.key_norm.scale"),
        (f"{prefix}{block_prefix}attn.norm_added_q.weight", f"double_blocks.{block_idx}.txt_attn.norm.query_norm.scale"),
        (f"{prefix}{block_prefix}attn.norm_added_k.weight", f"double_blocks.{block_idx}.txt_attn.norm.key_norm.scale")
    ]
    
    for src_key, target_key in norm_keys:
        target_dict, source_dict = convert_layer_weights(
            target_dict, source_dict, src_key, target_key
        )
    
    # Convert MLP weights
    mlp_keys = [
        (f"{prefix}{block_prefix}ff.net.0.proj.weight", f"double_blocks.{block_idx}.img_mlp.0.weight"),
        (f"{prefix}{block_prefix}ff.net.2.weight", f"double_blocks.{block_idx}.img_mlp.2.weight"),
        (f"{prefix}{block_prefix}ff_context.net.0.proj.weight", f"double_blocks.{block_idx}.txt_mlp.0.weight"),
        (f"{prefix}{block_prefix}ff_context.net.2.weight", f"double_blocks.{block_idx}.txt_mlp.2.weight")
    ]
    
    for src_key, target_key in mlp_keys:
        target_dict, source_dict = convert_layer_weights(
            target_dict, source_dict, src_key, target_key
        )
    
    # Convert output projections
    output_keys = [
        (f"{prefix}{block_prefix}attn.to_out.0.weight", f"double_blocks.{block_idx}.img_attn.proj.weight"),
        (f"{prefix}{block_prefix}attn.to_add_out.weight", f"double_blocks.{block_idx}.txt_attn.proj.weight")
    ]
    
    for src_key, target_key in output_keys:
        target_dict, source_dict = convert_layer_weights(
            target_dict, source_dict, src_key, target_key
        )
    
    return target_dict, source_dict


def convert_single_transformer_block(target_dict, source_dict, prefix, block_idx):
    """
    Convert weights for a single transformer block.
    
    Args:
        target_dict: Dictionary to store converted weights
        source_dict: Source dictionary containing weights
        prefix: Prefix for the keys in the state dictionary
        block_idx: Block index
        
    Returns:
        Tuple of (updated target_dict, updated source_dict)
    """
    block_prefix = f"single_transformer_blocks.{block_idx}."
    
    # Convert norm
    target_dict, source_dict = convert_layer_weights(
        target_dict,
        source_dict,
        f"{prefix}{block_prefix}norm.linear.weight",
        f"single_blocks.{block_idx}.modulation.lin.weight",
    )
    
    try:
        # Convert Q, K, V, MLP by concatenating
        q_A = source_dict.pop(f"{prefix}{block_prefix}attn.to_q.lora_A.weight")
        q_B = source_dict.pop(f"{prefix}{block_prefix}attn.to_q.lora_B.weight")
        k_A = source_dict.pop(f"{prefix}{block_prefix}attn.to_k.lora_A.weight")
        k_B = source_dict.pop(f"{prefix}{block_prefix}attn.to_k.lora_B.weight")
        v_A = source_dict.pop(f"{prefix}{block_prefix}attn.to_v.lora_A.weight")
        v_B = source_dict.pop(f"{prefix}{block_prefix}attn.to_v.lora_B.weight")
        mlp_A = source_dict.pop(f"{prefix}{block_prefix}proj_mlp.lora_A.weight")
        mlp_B = source_dict.pop(f"{prefix}{block_prefix}proj_mlp.lora_B.weight")
        
        target_dict[f"single_blocks.{block_idx}.linear1.lora_A.weight"] = torch.cat(
            [q_A, k_A, v_A, mlp_A], dim=0
        )
        target_dict[f"single_blocks.{block_idx}.linear1.lora_B.weight"] = torch.cat(
            [q_B, k_B, v_B, mlp_B], dim=0
        )
    except KeyError as e:
        print(f"Error processing attention weights for single block {block_idx}: {e}")
        raise
    
    # Convert output projection
    target_dict, source_dict = convert_layer_weights(
        target_dict,
        source_dict,
        f"{prefix}{block_prefix}proj_out.weight",
        f"single_blocks.{block_idx}.linear2.weight",
    )
    
    return target_dict, source_dict


def convert_embedding_layers(target_dict, source_dict, prefix, has_guidance=True):
    """
    Convert time, text, guidance, and context embedding layers.
    
    Args:
        target_dict: Dictionary to store converted weights
        source_dict: Source dictionary containing weights
        prefix: Prefix for the keys in the state dictionary
        has_guidance: Whether the model has guidance embedding
        
    Returns:
        Tuple of (updated target_dict, updated source_dict)
    """
    # Convert time embedding
    target_dict, source_dict = convert_layer_weights(
        target_dict,
        source_dict,
        f"{prefix}time_text_embed.timestep_embedder.linear_1.weight",
        "time_in.in_layer.weight",
    )
    
    # Convert text embedding
    text_embed_keys = [
        (f"{prefix}time_text_embed.text_embedder.linear_1.weight", "vector_in.in_layer.weight"),
        (f"{prefix}time_text_embed.text_embedder.linear_2.weight", "vector_in.out_layer.weight")
    ]
    
    for src_key, target_key in text_embed_keys:
        target_dict, source_dict = convert_layer_weights(
            target_dict, source_dict, src_key, target_key
        )
    
    # Convert guidance embedding if needed
    if has_guidance:
        guidance_keys = [
            (f"{prefix}time_text_embed.guidance_embedder.linear_1.weight", "guidance_in.in_layer.weight"),
            (f"{prefix}time_text_embed.guidance_embedder.linear_2.weight", "guidance_in.out_layer.weight")
        ]
        
        for src_key, target_key in guidance_keys:
            target_dict, source_dict = convert_layer_weights(
                target_dict, source_dict, src_key, target_key
            )
    
    # Convert context and image embedders
    embed_keys = [
        (f"{prefix}context_embedder.weight", "txt_in.weight"),
        (f"{prefix}x_embedder.weight", "img_in.weight")
    ]
    
    for src_key, target_key in embed_keys:
        target_dict, source_dict = convert_layer_weights(
            target_dict, source_dict, src_key, target_key
        )
    
    return target_dict, source_dict


def convert_output_layers(target_dict, source_dict, prefix):
    """
    Convert final output layers.
    
    Args:
        target_dict: Dictionary to store converted weights
        source_dict: Source dictionary containing weights
        prefix: Prefix for the keys in the state dictionary
        
    Returns:
        Tuple of (updated target_dict, updated source_dict)
    """
    output_keys = [
        (f"{prefix}proj_out.weight", "final_layer.linear.weight"),
        (f"{prefix}proj_out.bias", "final_layer.linear.bias"),
        (f"{prefix}norm_out.linear.weight", "final_layer.adaLN_modulation.1.weight")
    ]
    
    for src_key, target_key in output_keys:
        target_dict, source_dict = convert_layer_weights(
            target_dict, source_dict, src_key, target_key
        )
    
    return target_dict, source_dict


def convert_diffusers_to_flux_transformer_checkpoint(
    diffusers_state_dict,
    num_layers=19,
    num_single_layers=38,
    has_guidance=True,
    old_prefix="base_model.model.",
    new_prefix=FLUX_KOHYA_TRANSFORMER_KEY,
):
    """
    Convert a diffusers state dictionary to flux transformer checkpoint format.
    
    Args:
        diffusers_state_dict: Source diffusers state dictionary
        num_layers: Number of double transformer layers
        num_single_layers: Number of single transformer layers
        has_guidance: Whether the model has guidance embedding
        prefix: Prefix for keys in the source dictionary
        
    Returns:
        A new state dictionary in flux transformer format
    """
    # Create a new state dictionary
    flux_state_dict = {}
    
    # Convert embedding layers
    flux_state_dict, diffusers_state_dict = convert_embedding_layers(
        flux_state_dict, diffusers_state_dict, old_prefix, has_guidance
    )
    
    # Convert double transformer blocks
    for i in range(num_layers):
        flux_state_dict, diffusers_state_dict = convert_double_transformer_block(
            flux_state_dict, diffusers_state_dict, old_prefix, i
        )
    
    # Convert single transformer blocks
    for i in range(num_single_layers):
        flux_state_dict, diffusers_state_dict = convert_single_transformer_block(
            flux_state_dict, diffusers_state_dict, old_prefix, i
        )
    
    # Convert output layers
    flux_state_dict, diffusers_state_dict = convert_output_layers(
        flux_state_dict, diffusers_state_dict, old_prefix
    )
    
    # Check for leftover keys
    if diffusers_state_dict:
        print(f"Unexpected keys: {list(diffusers_state_dict.keys())}")
    
    # Replace the old prefix with the new prefix
    keys = list(flux_state_dict.keys())
    for key in keys:
        new_key = f"{new_prefix}.{key}"
        flux_state_dict[new_key] = flux_state_dict.pop(key)
    return flux_state_dict

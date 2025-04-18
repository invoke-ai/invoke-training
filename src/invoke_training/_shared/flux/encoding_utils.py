import torch
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

def get_clip_prompt_embeds(
    prompt: Union[str, List[str]],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
    num_images_per_prompt: int = 1,
    tokenizer_max_length: int = 77,
) -> torch.FloatTensor:
    """Encodes the prompt using CLIP text encoder and returns pooled embeddings."""
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    # Process text input with the tokenizer
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    
    # Check if truncation occurred
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
        print(f"Warning: The following part of your input was truncated: {removed_text}")

    # Get prompt embeddings through the text encoder
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # Duplicate text embeddings for each generation per prompt
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def get_t5_prompt_embeds(
    prompt: Union[str, List[str]],
    t5_tokenizer: T5TokenizerFast,
    t5_encoder: T5EncoderModel,
    device: torch.device,
    num_images_per_prompt: int = 1,
    tokenizer_max_length: int = 512
) -> torch.FloatTensor:
    """Encodes the prompt using T5 text encoder."""
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    # Process text input with the tokenizer
    text_inputs = t5_tokenizer(
        prompt,
        padding="max_length",
        max_length=t5_tokenizer_max_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

    # Check if truncation occurred
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer_2.batch_decode(untruncated_ids[:, t5_tokenizer_max_length - 1 : -1])
        print(f"Warning: The following part of your input was truncated: {removed_text}")

    # Get prompt embeddings through the text encoder
    prompt_embeds = text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

    dtype = text_encoder_2.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # Get shape and duplicate for multiple generations
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def handle_lora_scale(
    text_encoder: CLIPTextModel,
    text_encoder_2: T5EncoderModel,
    lora_scale: Optional[float] = None,
    use_peft_backend: bool = False
):
    """Handles LoRA scale adjustments for text encoders."""
    if lora_scale is not None and use_peft_backend:
        from peft.utils import scale_lora_layers, unscale_lora_layers
        
        # Apply LoRA scaling to text encoders if they exist
        if text_encoder is not None:
            scale_lora_layers(text_encoder, lora_scale)
        if text_encoder_2 is not None:
            scale_lora_layers(text_encoder_2, lora_scale)
            
        return True
    return False


def reset_lora_scale(
    text_encoder: CLIPTextModel,
    text_encoder_2: T5EncoderModel,
    lora_scale: Optional[float] = None,
    lora_applied: bool = False,
    use_peft_backend: bool = False
):
    """Resets LoRA scale for text encoders if it was applied."""
    if lora_applied and use_peft_backend:
        from peft.utils import unscale_lora_layers
        
        # Reset LoRA scaling
        if text_encoder is not None:
            unscale_lora_layers(text_encoder, lora_scale)
        if text_encoder_2 is not None:
            unscale_lora_layers(text_encoder_2, lora_scale)

# A lot of this code was adapted from:
# https://github.com/huggingface/diffusers/blob/ea81a4228d8ff16042c3ccaf61f0e588e60166cd/src/diffusers/pipelines/flux/pipeline_flux.py#L310-L387
def encode_prompt(
    prompt: Union[str, List[str]],
    prompt_2: Optional[Union[str, List[str]]],
    clip_tokenizer: CLIPTokenizer,
    t5_tokenizer: T5TokenizerFast,
    clip_encoder: CLIPTextModel,
    t5_encoder: T5EncoderModel,
    device: torch.device,
    num_images_per_prompt: int = 1,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    lora_scale: Optional[float] = None,
    use_peft_backend: bool = False,
    clip_tokenizer_max_length: int = 77,
    t5_tokenizer_max_length: int = 512
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Encodes the prompt using both CLIP and T5 text encoders.
    
    Returns:
        Tuple containing:
            - T5 text embeddings
            - CLIP pooled embeddings
            - Text IDs
    """
    # Apply LoRA scale if needed
    lora_applied = handle_lora_scale(
        text_encoder, text_encoder_2, lora_scale, use_peft_backend
    )
    
    # If no pre-generated embeddings, create them
    if prompt_embeds is None:
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # Get CLIP pooled embeddings
        pooled_prompt_embeds = get_clip_prompt_embeds(
            prompt=prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer_max_length=clip_tokenizer_max_length
        )
        
        # Get T5 text embeddings
        prompt_embeds = get_t5_prompt_embeds(
            prompt=prompt_2,
            t5_tokenizer=t5_tokenizer,
            t5_encoder=t5_encoder,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer_max_length=t5_tokenizer_max_length
        )
    
    # Reset LoRA scale if it was applied
    reset_lora_scale(
        text_encoder, text_encoder_2, lora_scale, lora_applied, use_peft_backend
    )
    
    # Create text_ids placeholder for model
    dtype = text_encoder.dtype if text_encoder is not None else text_encoder_2.dtype
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    
    return prompt_embeds, pooled_prompt_embeds, text_ids
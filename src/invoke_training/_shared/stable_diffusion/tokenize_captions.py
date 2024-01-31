import torch
from transformers import CLIPTokenizer

from invoke_training._shared.stable_diffusion.textual_inversion import expand_placeholders_in_caption


def tokenize_captions(tokenizer: CLIPTokenizer, captions: list[str]) -> torch.Tensor:
    """Tokenize a list of caption.

    Args:
        tokenizer (CLIPTokenizer): The tokenizer.
        caption (str): The caption.

    Returns:
        torch.Tensor: The token IDs.
    """
    caption_token_ids = []
    for caption in captions:
        caption = expand_placeholders_in_caption(caption, tokenizer)
        input = tokenizer(
            caption,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        caption_token_ids.append(input.input_ids[0, ...])

    caption_token_ids = torch.stack(caption_token_ids)
    return caption_token_ids

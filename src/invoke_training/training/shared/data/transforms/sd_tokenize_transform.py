import typing

from transformers import CLIPTokenizer


def tokenize_caption(tokenizer: CLIPTokenizer, caption: str):
    """Tokenize a caption for Stable Diffusion v1/v2 training.

    Args:
        tokenizer (CLIPTokenizer): The tokenizer.
        caption (str): The caption.

    Returns:
        torch.Tensor: The token IDs.
    """
    input = tokenizer(
        caption,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return input.input_ids[0, ...]


class SDTokenizeTransform:
    """A transform that tokenizes captions for Stable Diffusion v1/v2 training."""

    def __init__(self, tokenizer: CLIPTokenizer):
        self._tokenizer = tokenizer

    def __call__(self, data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        data["caption_token_ids"] = tokenize_caption(self._tokenizer, data["caption"])
        return data

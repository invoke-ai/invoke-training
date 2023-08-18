import typing

from transformers import CLIPTokenizer


def tokenize_caption(tokenizer: CLIPTokenizer, caption: str):
    """Tokenize a caption.

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
    """A transform that tokenizes captions for Stable Diffusion training."""

    def __init__(
        self, tokenizer: CLIPTokenizer, src_caption_key: str = "caption", dst_token_key: str = "caption_token_ids"
    ):
        self._tokenizer = tokenizer
        self._src_caption_key = src_caption_key
        self._dst_token_key = dst_token_key

    def __call__(self, data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        data[self._dst_token_key] = tokenize_caption(self._tokenizer, data[self._src_caption_key])
        return data

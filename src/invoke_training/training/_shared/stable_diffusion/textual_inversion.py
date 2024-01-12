from transformers import PreTrainedTokenizer


def add_tokens_to_tokenizer(placeholder_tokens: list[str], tokenizer: PreTrainedTokenizer):
    """Add new tokens to a tokenizer.

    Raises:
        ValueError: Raises if the tokenizer already contains one of the tokens in `placeholder_tokens`.
    """
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != len(placeholder_tokens):
        raise ValueError(
            f"The tokenizer already contains one of the tokens in '{placeholder_tokens}'. Please pass a different"
            " 'placeholder_token' that is not already in the tokenizer."
        )

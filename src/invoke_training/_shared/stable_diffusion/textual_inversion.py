import logging

import torch
from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer, PreTrainedTokenizer

from invoke_training._shared.checkpoints.serialization import load_state_dict


def _expand_placeholder_token(placeholder_token: str, num_vectors: int = 1) -> list[str]:
    """Expand a placeholder token into a list of placeholder tokens based on the number of embedding vectors being
    trained.
    """
    placeholder_tokens = [placeholder_token]
    if num_vectors < 1:
        raise ValueError(f"num_vectors must be >1, but is '{num_vectors}'.")
    # Add dummy placeholder tokens if num_vectors > 1.
    for i in range(1, num_vectors):
        placeholder_tokens.append(f"{placeholder_token}_{i}")
    return placeholder_tokens


def _add_tokens_to_tokenizer(placeholder_tokens: list[str], tokenizer: PreTrainedTokenizer):
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


def expand_placeholders_in_caption(caption: str, tokenizer: CLIPTokenizer) -> str:
    """Expand any multi-vector placeholder tokens in the caption.

    For example, "a dog in the style of my_placeholder", could get expanded to "a dog in the style of my_placeholder
    my_placeholder_1 my_placeholder_2".

    This implementation is based on
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/textual_inversion.py#L144. This logic gets
    applied automatically when running a full diffusers text-to-image pipeline.
    """
    tokens = tokenizer.tokenize(caption)
    unique_tokens = set(tokens)
    for token in unique_tokens:
        if token in tokenizer.added_tokens_encoder:
            replacement = token
            i = 1
            while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                replacement += f" {token}_{i}"
                i += 1

            if replacement != token:
                # If the replacement is different from the original token, then we double check that the replacement
                # isn't already in the caption. If the replacement is already in the caption, this probably means that
                # someone didn't realize that placeholder expansion is handled here.
                assert replacement not in caption

            caption = caption.replace(token, replacement)

    return caption


def initialize_placeholder_tokens_from_initializer_token(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    initializer_token: str,
    placeholder_token: str,
    num_vectors: int,
    logger: logging.Logger,
) -> tuple[list[str], list[int]]:
    # Convert the initializer_token to a token id.
    initializer_token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    if len(initializer_token_ids) > 1:
        logger.warning(
            f"The initializer_token '{initializer_token}' gets tokenized to {len(initializer_token_ids)} tokens. "
            "Only the first token will be used. It is recommended to choose a different initializer_token that maps to "
            "a single token."
        )

    initializer_token_id = initializer_token_ids[0]

    # Expand the tokenizer / text_encoder to include the placeholder tokens.
    placeholder_tokens = _expand_placeholder_token(placeholder_token, num_vectors=num_vectors)
    _add_tokens_to_tokenizer(placeholder_tokens, tokenizer)
    # Resize the token embeddings as we have added new special tokens to the tokenizer.
    text_encoder.resize_token_embeddings(len(tokenizer))
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    # convert_tokens_to_ids returns a `int | list[int]` type, but since we pass in a list it should always return a
    # list.
    assert isinstance(placeholder_token_ids, list)

    # Initialize the newly-added placeholder token(s) with the embeddings of the initializer token.
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for placeholder_token_id in placeholder_token_ids:
            token_embeds[placeholder_token_id] = token_embeds[initializer_token_id].clone()

    return placeholder_tokens, placeholder_token_ids


def initialize_placeholder_tokens_from_initial_phrase(
    tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, initial_phrase: str, placeholder_token: str
) -> tuple[list[str], list[int]]:
    # Convert the initial_phrase to token ids.
    initial_token_ids = tokenizer.encode(initial_phrase, add_special_tokens=False)

    # Expand the tokenizer / text_encoder to include one placeholder token for each token in the initial_phrase.
    placeholder_tokens = _expand_placeholder_token(placeholder_token, num_vectors=len(initial_token_ids))
    _add_tokens_to_tokenizer(placeholder_tokens, tokenizer)
    # Resize the token embeddings as we have added new special tokens to the tokenizer.
    text_encoder.resize_token_embeddings(len(tokenizer))
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    # convert_tokens_to_ids returns a `int | list[int]` type, but since we pass in a list it should always return a
    # list.
    assert isinstance(placeholder_token_ids, list)

    # Initialize the newly-added placeholder token(s) with the embeddings of the initial phrase.
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for placeholder_token_id, initial_token_id in zip(placeholder_token_ids, initial_token_ids):
            token_embeds[placeholder_token_id] = token_embeds[initial_token_id].clone()

    return placeholder_tokens, placeholder_token_ids


def initialize_placeholder_tokens_from_initial_embedding(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    initial_embedding_file: str,
    placeholder_token: str,
    num_vectors: int,
) -> tuple[list[str], list[int]]:
    # Expand the tokenizer / text_encoder to include the placeholder tokens.
    placeholder_tokens = _expand_placeholder_token(placeholder_token, num_vectors=num_vectors)
    _add_tokens_to_tokenizer(placeholder_tokens, tokenizer)
    # Resize the token embeddings as we have added new special tokens to the tokenizer.
    text_encoder.resize_token_embeddings(len(tokenizer))

    state_dict = load_state_dict(initial_embedding_file)
    if placeholder_token not in state_dict:
        raise ValueError(
            f"The initial embedding at '{initial_embedding_file}' does not contain an embedding for placeholder token "
            f"'{placeholder_token}'."
        )

    embeddings = state_dict[placeholder_token]
    if embeddings.shape[0] != len(placeholder_tokens):
        raise ValueError(
            f"The number of initial embeddings in '{initial_embedding_file}' ({embeddings.shape[0]}) does not match "
            f"the expected number of placeholder tokens ({len(placeholder_tokens)})."
        )

    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    # convert_tokens_to_ids returns a `int | list[int]` type, but since we pass in a list it should always return a
    # list.
    assert isinstance(placeholder_token_ids, list)

    # Initialize the newly-added placeholder token(s) with the loaded embeddings.
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for i, token_id in enumerate(placeholder_token_ids):
            token_embeds[token_id] = embeddings[i].clone()

    return placeholder_tokens, placeholder_token_ids


def restore_original_embeddings(
    tokenizer: CLIPTokenizer,
    placeholder_token_ids: list[int],
    accelerator: Accelerator,
    text_encoder: CLIPTextModel,
    orig_embeds_params: torch.Tensor,
):
    """Restore the text_encoder embeddings that we are not actively training to make sure they don't change.

    TODO(ryand): Look into whether this is actually necessary if we set requires_grad correctly.
    """
    index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
    index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False
    index_updates = ~index_no_updates
    with torch.no_grad():
        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
        unwrapped_text_encoder.get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]

        target_std = unwrapped_text_encoder.get_input_embeddings().weight[index_no_updates].std()
        new_embeddings = unwrapped_text_encoder.get_input_embeddings().weight[index_updates]
        target_over_new_std = target_std / new_embeddings.std()

        # Scale the new embeddings towards the target embeddings. Raise to the 0.1 power to avoid large changes.
        new_embeddings = new_embeddings * (target_over_new_std**0.1)
        unwrapped_text_encoder.get_input_embeddings().weight[index_updates] = new_embeddings

import torch
from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer, PreTrainedTokenizer

from invoke_training.training._shared.checkpoints.serialization import load_state_dict


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


def initialize_placeholder_tokens_from_initializer_token(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    initializer_token: str,
    placeholder_token: str,
    num_vectors: int,
) -> tuple[list[str], list[int]]:
    # Convert the initializer_token to a token id.
    initializer_token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    if len(initializer_token_ids) > 1:
        raise ValueError(
            f"The initializer_token '{initializer_token}' gets tokenized to {len(initializer_token_ids)} tokens."
            " Choose a different initializer that maps to a single token."
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
    with torch.no_grad():
        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[index_no_updates] = orig_embeds_params[
            index_no_updates
        ]

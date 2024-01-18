from pathlib import Path

import pytest
import torch

from invoke_training.training._shared.stable_diffusion.model_loading_utils import load_models_sd
from invoke_training.training._shared.stable_diffusion.textual_inversion import (
    _expand_placeholder_token,
    initialize_placeholder_tokens_from_initial_embedding,
    initialize_placeholder_tokens_from_initial_phrase,
    initialize_placeholder_tokens_from_initializer_token,
)

from .ti_embedding_checkpoint_fixture import sdv1_embedding_path  # noqa: F401


@pytest.mark.parametrize(
    ["placeholder_token", "num_vectors", "expected_placeholder_tokens"],
    [("abc", 1, ["abc"]), ("abc", 2, ["abc", "abc_1"]), ("abc", 3, ["abc", "abc_1", "abc_2"])],
)
def test_expand_placeholder_token(placeholder_token: str, num_vectors: int, expected_placeholder_tokens: list[str]):
    assert _expand_placeholder_token(placeholder_token, num_vectors) == expected_placeholder_tokens


def test_expand_placeholder_token_raises_on_invalid_num_vectors():
    with pytest.raises(ValueError):
        _expand_placeholder_token("abc", 0)


@pytest.mark.loads_model
def test_initialize_placeholder_tokens_from_initializer_token():
    tokenizer, noise_scheduler, text_encoder, vae, unet = load_models_sd(
        model_name_or_path="runwayml/stable-diffusion-v1-5", hf_variant="fp16"
    )

    initializer_token = "dog"
    num_vectors = 2
    placeholder_tokens, placeholder_token_ids = initialize_placeholder_tokens_from_initializer_token(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        initializer_token=initializer_token,
        placeholder_token="dog_placeholder",
        num_vectors=num_vectors,
    )

    assert len(placeholder_tokens) == num_vectors
    assert len(placeholder_token_ids) == num_vectors
    assert placeholder_tokens == ["dog_placeholder", "dog_placeholder_1"]

    token_embeds = text_encoder.get_input_embeddings().weight.data
    initializer_token_id = tokenizer.encode(initializer_token, add_special_tokens=False)[0]
    with torch.no_grad():
        for placeholder_token_id in placeholder_token_ids:
            assert torch.allclose(token_embeds[placeholder_token_id], token_embeds[initializer_token_id])


@pytest.mark.loads_model
def test_initialize_placeholder_tokens_from_initial_phrase():
    tokenizer, noise_scheduler, text_encoder, vae, unet = load_models_sd(
        model_name_or_path="runwayml/stable-diffusion-v1-5", hf_variant="fp16"
    )

    initial_phrase = "little brown dog"
    placeholder_tokens, placeholder_token_ids = initialize_placeholder_tokens_from_initial_phrase(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        initial_phrase=initial_phrase,
        placeholder_token="dog_placeholder",
    )

    expected_num_vectors = 3
    assert len(placeholder_tokens) == expected_num_vectors
    assert len(placeholder_token_ids) == expected_num_vectors
    assert placeholder_tokens == ["dog_placeholder", "dog_placeholder_1", "dog_placeholder_2"]

    token_embeds = text_encoder.get_input_embeddings().weight.data
    initial_token_ids = tokenizer.encode(initial_phrase, add_special_tokens=False)
    assert len(initial_token_ids) == expected_num_vectors
    with torch.no_grad():
        for placeholder_token_id, initial_token_id in zip(placeholder_token_ids, initial_token_ids):
            assert torch.allclose(token_embeds[placeholder_token_id], token_embeds[initial_token_id])


@pytest.mark.loads_model
def test_initialize_placeholder_tokens_from_initial_embedding(sdv1_embedding_path: Path):  # noqa: F811
    tokenizer, noise_scheduler, text_encoder, vae, unet = load_models_sd(
        model_name_or_path="runwayml/stable-diffusion-v1-5", hf_variant="fp16"
    )

    placeholder_token = "custom_token"
    num_vectors = 2
    placeholder_tokens, placeholder_token_ids = initialize_placeholder_tokens_from_initial_embedding(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        initial_embedding_file=str(sdv1_embedding_path),
        placeholder_token=placeholder_token,
        num_vectors=num_vectors,
    )

    assert len(placeholder_tokens) == num_vectors
    assert len(placeholder_token_ids) == num_vectors
    assert placeholder_tokens == ["custom_token", "custom_token_1"]

    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for placeholder_token_id in placeholder_token_ids:
            # The placeholder embeddings should be initialized to zero, because this is how they are initialized in the
            # dummy sdv1_embedding_path checkpoint.
            assert torch.allclose(
                token_embeds[placeholder_token_id], torch.zeros_like(token_embeds[placeholder_token_id])
            )

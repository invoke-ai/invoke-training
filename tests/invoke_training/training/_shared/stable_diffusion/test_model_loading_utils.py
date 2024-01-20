from pathlib import Path

import pytest
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from invoke_training.training._shared.stable_diffusion.model_loading_utils import load_models_sd, load_models_sdxl

from .ti_embedding_checkpoint_fixture import sdv1_embedding_path, sdxl_embedding_path  # noqa: F401


@pytest.mark.loads_model
def test_load_models_sd(sdv1_embedding_path):  # noqa: F811
    model_name = "runwayml/stable-diffusion-v1-5"

    tokenizer, noise_scheduler, text_encoder, vae, unet = load_models_sd(
        model_name_or_path=model_name,
        hf_variant="fp16",
        base_embeddings={"special_test_token": str(sdv1_embedding_path)},
    )

    token_ids = tokenizer.encode("special_test_token special_test_token_1", add_special_tokens=False)
    assert len(token_ids) == 2

    token_embeds = text_encoder.get_input_embeddings().weight.data
    for token_id in token_ids:
        # The embedding should be all zeros, because that is how it was initialized in the sdv1_embedding_path
        # fixture.
        assert torch.allclose(token_embeds[token_id], torch.zeros_like(token_embeds[token_id]))


@pytest.mark.loads_model
def test_load_models_sdxl(sdxl_embedding_path: Path):  # noqa: F811
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"

    tokenizer_1, tokenizer_2, noise_scheduler, text_encoder_1, text_encoder_2, vae, unet = load_models_sdxl(
        model_name_or_path=model_name,
        hf_variant="fp16",
        base_embeddings={"special_test_token": str(sdxl_embedding_path)},
    )

    # Validate that the embeddings were applied correctly.
    def validate_ti_embeddings(tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel):
        token_ids = tokenizer.encode("special_test_token special_test_token_1", add_special_tokens=False)
        assert len(token_ids) == 2

        token_embeds = text_encoder.get_input_embeddings().weight.data
        for token_id in token_ids:
            # The embedding should be all zeros, because that is how it was initialized in the sdxl_embedding_path
            # fixture.
            assert torch.allclose(token_embeds[token_id], torch.zeros_like(token_embeds[token_id]))

    validate_ti_embeddings(tokenizer_1, text_encoder_1)
    validate_ti_embeddings(tokenizer_2, text_encoder_2)

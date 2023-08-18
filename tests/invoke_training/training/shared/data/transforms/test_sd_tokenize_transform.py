import pytest
import torch
from transformers import CLIPTokenizer

from invoke_training.training.shared.data.transforms.sd_tokenize_transform import (
    SDTokenizeTransform,
)


@pytest.mark.loads_model
def test_sd_tokenize_transform():
    """Test that SDTokenizeTransform produce a tensor of the expected shape."""
    # Load CLIPTokenizer.
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer",
        local_files_only=True,
        revision="c9ab35ff5f2c362e9e22fbafe278077e196057f0",
    )

    tf = SDTokenizeTransform(tokenizer)
    in_example = {"caption": "This is a test caption."}

    out_example = tf(in_example)

    caption_token_ids = out_example["caption_token_ids"]
    assert isinstance(caption_token_ids, torch.Tensor)
    assert caption_token_ids.dtype == torch.int64
    assert caption_token_ids.shape == (tokenizer.model_max_length,)

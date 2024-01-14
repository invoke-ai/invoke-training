import pytest

from invoke_training.training._shared.stable_diffusion.textual_inversion import expand_placeholder_token


@pytest.mark.parametrize(
    ["placeholder_token", "num_vectors", "expected_placeholder_tokens"],
    [("abc", 1, ["abc"]), ("abc", 2, ["abc", "abc_1"]), ("abc", 3, ["abc", "abc_1", "abc_2"])],
)
def test_expand_placeholder_token(placeholder_token: str, num_vectors: int, expected_placeholder_tokens: list[str]):
    assert expand_placeholder_token(placeholder_token, num_vectors) == expected_placeholder_tokens


def test_expand_placeholder_token_raises_on_invalid_num_vectors():
    with pytest.raises(ValueError):
        expand_placeholder_token("abc", 0)

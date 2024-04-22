import pytest

from invoke_training.ui.utils.prompts import (
    convert_pos_neg_prompts_to_ui_prompts,
    convert_ui_prompts_to_pos_neg_prompts,
    split_pos_neg_prompts,
)


@pytest.mark.parametrize(
    ["prompt", "expected_positive_prompt", "expected_negative_prompt"],
    [
        # Simple positive and negative prompt.
        ("positive prompt[NEG]negative prompt", "positive prompt", "negative prompt"),
        # Positive prompt with no negative prompt.
        ("positive prompt", "positive prompt", ""),
        # Empty prompt.
        ("", "", ""),
    ],
)
def test_split_pos_neg_prompts(prompt: str, expected_positive_prompt: str, expected_negative_prompt: str):
    positive_prompt, negative_prompt = split_pos_neg_prompts(prompt)
    assert positive_prompt == expected_positive_prompt
    assert negative_prompt == expected_negative_prompt


@pytest.mark.parametrize(
    "prompt",
    [
        # Multiple negative prompt delimiters.
        "positive prompt[NEG]negative prompt[NEG]negative prompt",
    ],
)
def test_split_pos_neg_prompts_raises_value_error(prompt: str):
    with pytest.raises(ValueError):
        split_pos_neg_prompts(prompt)


# Test cases for conversion between UI prompts and positive/negative prompts.
# Each test case consists of: (ui_prompts, positive_prompts, negative_prompts)
prompt_conversion_test_cases = [
    # Positive prompts.
    (
        "positive prompt 1\npositive prompt 2\npositive prompt 3",
        ["positive prompt 1", "positive prompt 2", "positive prompt 3"],
        None,
    ),
    # Positive prompts with trailing \n.
    (
        "positive prompt 1\npositive prompt 2\npositive prompt 3\n",
        ["positive prompt 1", "positive prompt 2", "positive prompt 3"],
        None,
    ),
    # Positive and negative prompts.
    (
        "positive prompt 1[NEG]negative prompt 1\npositive prompt 2[NEG]negative prompt 2\n"
        "positive prompt 3[NEG]negative prompt 3\n",
        ["positive prompt 1", "positive prompt 2", "positive prompt 3"],
        ["negative prompt 1", "negative prompt 2", "negative prompt 3"],
    ),
    # Some missing negative prompts.
    (
        "positive prompt 1[NEG]negative prompt 1\npositive prompt 2\npositive prompt 3[NEG]negative prompt 3\n",
        ["positive prompt 1", "positive prompt 2", "positive prompt 3"],
        ["negative prompt 1", "", "negative prompt 3"],
    ),
]


@pytest.mark.parametrize(
    ["ui_prompts", "expected_positive_prompts", "expected_negative_prompts"], prompt_conversion_test_cases
)
def test_convert_ui_prompts_to_pos_neg_prompts(
    ui_prompts: str, expected_positive_prompts: list[str], expected_negative_prompts: list[str | None] | None
):
    positive_prompts, negative_prompts = convert_ui_prompts_to_pos_neg_prompts(ui_prompts)
    assert positive_prompts == expected_positive_prompts
    assert negative_prompts == expected_negative_prompts


@pytest.mark.parametrize(["expected_ui_prompts", "positive_prompts", "negative_prompts"], prompt_conversion_test_cases)
def test_convert_pos_neg_prompts_to_ui_prompts(
    expected_ui_prompts: str, positive_prompts: list[str], negative_prompts: list[str | None] | None
):
    ui_prompts = convert_pos_neg_prompts_to_ui_prompts(positive_prompts, negative_prompts)
    assert ui_prompts == expected_ui_prompts.strip()

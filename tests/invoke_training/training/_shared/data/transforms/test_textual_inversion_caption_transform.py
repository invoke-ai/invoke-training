import pytest

from invoke_training.training._shared.data.transforms.textual_inversion_caption_transform import (
    TextualInversionCaptionTransform,
)


def test_textual_inversion_caption_transform():
    tf = TextualInversionCaptionTransform(
        field_name="test_field", placeholder_str="placeholder", caption_templates=["template 1 {}"]
    )

    in_example = {"existing": 2}

    out_example = tf(in_example)

    assert out_example == {"existing": 2, "test_field": "template 1 placeholder"}


def test_textual_inversion_caption_transform_seed():
    field_name = "test_field"
    placeholder_str = "placeholder"
    caption_templates = ["template 1 {}", "template 2 {}"]
    tf = TextualInversionCaptionTransform(
        field_name=field_name,
        placeholder_str=placeholder_str,
        caption_templates=caption_templates,
        seed=123,
    )

    # Run on 10 examples with baseline seed 123.
    out_examples = [tf({}) for _ in range(10)]

    # Run on 10 examples with same seed and assert that results match.
    tf = TextualInversionCaptionTransform(
        field_name=field_name,
        placeholder_str=placeholder_str,
        caption_templates=caption_templates,
        seed=123,
    )
    out_examples_same_seed = [tf({}) for _ in range(10)]
    assert out_examples == out_examples_same_seed

    # Run on 10 examples with a different seed and assert that the results don't match.
    tf = TextualInversionCaptionTransform(
        field_name=field_name,
        placeholder_str=placeholder_str,
        caption_templates=caption_templates,
        seed=456,
    )
    out_examples_diff_seed = [tf({}) for _ in range(10)]
    assert out_examples != out_examples_diff_seed


def test_textual_inversion_caption_transform_bad_templates():
    tf = TextualInversionCaptionTransform(
        field_name="test_field", placeholder_str="placeholder", caption_templates=["template 1"]
    )

    in_example = {"existing": 2}

    with pytest.raises(AssertionError):
        _ = tf(in_example)

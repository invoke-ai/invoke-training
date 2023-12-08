from invoke_training.training.shared.data.transforms.shuffle_caption_transform import ShuffleCaptionTransform


def test_shuffle_caption_transform():
    tf = ShuffleCaptionTransform(field_name="test_field", seed=3)

    in_example = {"test_field": "prompt part 1, prompt part 2"}

    out_example = tf(in_example)

    # Note that the expected output depends on the seed.
    assert out_example == {"test_field": "prompt part 2, prompt part 1"}


def test_shuffle_caption_transform_no_delimiter():
    tf = ShuffleCaptionTransform(field_name="test_field")

    in_example = {"test_field": "prompt part 1"}

    out_example = tf(in_example)

    assert out_example == {"test_field": "prompt part 1"}

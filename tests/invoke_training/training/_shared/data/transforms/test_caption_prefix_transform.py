from invoke_training.training._shared.data.transforms.caption_prefix_transform import CaptionPrefixTransform


def test_caption_prefix_transform():
    tf = CaptionPrefixTransform(caption_field_name="caption", prefix="prefix ")

    in_example = {"caption": "original caption", "other": 2}

    out_example = tf(in_example)

    assert out_example == {"caption": "prefix original caption", "other": 2}

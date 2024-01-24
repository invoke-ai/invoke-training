from invoke_training.training._shared.data.transforms.concat_fields_transform import ConcatFieldsTransform


def test_caption_prefix_transform():
    tf = ConcatFieldsTransform(src_field_names=["caption", "caption_2"], dst_field_name="caption", separator=", ")

    in_example = {"caption": "original caption", "caption_2": "another caption", "other": 2}

    out_example = tf(in_example)

    assert out_example == {"caption": "original caption, another caption", "caption_2": "another caption", "other": 2}

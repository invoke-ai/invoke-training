from invoke_training.training._shared.data.transforms.drop_field_transform import DropFieldTransform


def test_drop_field_transform():
    tf = DropFieldTransform("drop")

    in_example = {"keep": 1, "drop": 2}

    out_example = tf(in_example)

    assert out_example == {"keep": 1}

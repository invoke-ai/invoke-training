from invoke_training.training.shared.data.transforms.constant_field_transform import (
    ConstantFieldTransform,
)


def test_constant_field_transform():
    tf = ConstantFieldTransform("test_field", 1)

    in_example = {"existing": 2}

    out_example = tf(in_example)

    assert out_example == {"existing": 2, "test_field": 1}

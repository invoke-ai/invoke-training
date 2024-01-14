import numpy as np
import pytest
from PIL import Image

from invoke_training.training._shared.data.utils.resize import resize_to_cover
from invoke_training.training._shared.data.utils.resolution import Resolution


@pytest.mark.parametrize(
    ["in_resolution", "size_to_cover", "expected_resolution"],
    [
        # Perfect match, no resize necessary.
        (Resolution(512, 768), Resolution(512, 768), Resolution(512, 768)),
        # Height matches, width covers, no resize necessary.
        (Resolution(768, 768), Resolution(768, 512), Resolution(768, 768)),
        # Width matches, height covers, no resize necessary.
        (Resolution(768, 768), Resolution(512, 768), Resolution(768, 768)),
        # Height matches, width does not cover, scale up.
        (Resolution(768, 256), Resolution(768, 512), Resolution(1536, 512)),
        # Width matches, height does not cover, scale up.
        (Resolution(256, 768), Resolution(512, 768), Resolution(512, 1536)),
        # Both width and height exceed target, scale down, limited by height.
        (Resolution(1024, 768), Resolution(768, 512), Resolution(768, 576)),
        # Both width and height exceed target, scale down, limited by width.
        (Resolution(768, 1024), Resolution(512, 768), Resolution(576, 768)),
    ],
)
def test_resize_to_cover(in_resolution: Resolution, size_to_cover: Resolution, expected_resolution: Resolution):
    in_img = np.zeros((in_resolution.height, in_resolution.width, 3), dtype=np.uint8)
    in_img = Image.fromarray(in_img)

    out_img = resize_to_cover(in_img, size_to_cover)

    assert out_img.height == expected_resolution.height
    assert out_img.width == expected_resolution.width

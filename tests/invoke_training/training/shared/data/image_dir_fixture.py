import numpy as np
import PIL.Image
import pytest


@pytest.fixture(scope="session")
def image_dir(tmp_path_factory: pytest.TempPathFactory):
    """A fixture that populates a temp directory with some test images and returns the directory path.

    Note that the 'session' scope is used to share the same directory across all tests in a session, because it is
    costly to populate the directory.

    Refer to https://docs.pytest.org/en/7.4.x/how-to/tmp_path.html#the-tmp-path-factory-fixture for details on the use
    of tmp_path_factory.
    """
    tmp_dir = tmp_path_factory.mktemp("dataset")

    for i in range(5):
        rgb_np = np.ones((128, 128, 3), dtype=np.uint8)
        rgb_pil = PIL.Image.fromarray(rgb_np)
        rgb_pil.save(tmp_dir / f"{i}.jpg")

    return tmp_dir

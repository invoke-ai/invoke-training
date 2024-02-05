import json

import numpy as np
import PIL.Image
import pytest

from invoke_training._shared.data.datasets.image_pair_preference_dataset import ImagePairPreferenceDataset


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


@pytest.fixture(scope="session")
def image_caption_dir(tmp_path_factory: pytest.TempPathFactory):
    """A fixture that populates a temp directory with some test images and caption files and returns the directory path.

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

        with open(tmp_dir / f"{i}.txt", "w") as f:
            f.write(f"caption {i}")

    return tmp_dir


@pytest.fixture(scope="session")
def image_caption_jsonl(tmp_path_factory: pytest.TempPathFactory):
    """A fixture that populates a temp directory with a ImageCaptionJsonlDataset and returns the jsonl file path.

    Note that the 'session' scope is used to share the same directory across all tests in a session, because it is
    costly to populate the directory.

    Refer to https://docs.pytest.org/en/7.4.x/how-to/tmp_path.html#the-tmp-path-factory-fixture for details on the use
    of tmp_path_factory.
    """
    tmp_dir = tmp_path_factory.mktemp("dataset")

    data = []

    for i in range(5):
        rgb_np = np.ones((128, 128, 3), dtype=np.uint8)
        rgb_pil = PIL.Image.fromarray(rgb_np)
        rgb_rel_path = f"{i}.jpg"
        rgb_pil.save(tmp_dir / rgb_rel_path)

        data.append({"image": str(rgb_rel_path), "text": f"caption {i}"})

    data_jsonl_path = tmp_dir / "data.jsonl"
    with open(data_jsonl_path, "w") as f:
        for d in data:
            json.dump(d, f)
            f.write("\n")

    return data_jsonl_path


@pytest.fixture(scope="session")
def image_pair_preference_dir(tmp_path_factory: pytest.TempPathFactory):
    """A fixture that populates a temp directory with a mock dataset intended to be consumed by
    ImagePairPreferenceDataset, and returns the directory path.

    Note that the 'session' scope is used to share the same directory across all tests in a session, because it is
    costly to populate the directory.

    Refer to https://docs.pytest.org/en/7.4.x/how-to/tmp_path.html#the-tmp-path-factory-fixture for details on the use
    of tmp_path_factory.
    """
    tmp_dir = tmp_path_factory.mktemp("dataset")

    prompts = ["mock prompt 1", "mock prompt 2"]
    metadata = []

    for prompt_idx in range(len(prompts)):
        for set_idx in range(3):
            set_dir = tmp_dir / f"prompt-{prompt_idx:0>4}" / f"set-{set_idx:0>4}"
            set_dir.mkdir(parents=True)
            set_metadata_dict = {"prompt": prompts[prompt_idx]}
            for image_idx in range(2):
                rgb_np = np.ones((32, 32, 3), dtype=np.uint8)
                rgb_pil = PIL.Image.fromarray(rgb_np)
                image_path = set_dir / f"image-{image_idx}.jpg"
                rgb_pil.save(image_path)
                set_metadata_dict[f"image_{image_idx}"] = str(image_path.relative_to(tmp_dir))
                set_metadata_dict[f"prefer_{image_idx}"] = image_idx == 0  # Always prefer image 0.

            metadata.append(set_metadata_dict)

    ImagePairPreferenceDataset.save_metadata(metadata=metadata, dataset_dir=tmp_dir)

    return tmp_dir

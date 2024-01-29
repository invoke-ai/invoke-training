import PIL.Image

from invoke_training._shared.data.datasets.image_pair_preference_dataset import ImagePairPreferenceDataset

from ..image_dir_fixture import image_pair_preference_dir  # noqa: F401


def test_image_dir_dataset_len(image_pair_preference_dir):  # noqa: F811
    dataset = ImagePairPreferenceDataset(str(image_pair_preference_dir))

    assert len(dataset) == 6


def test_image_dir_dataset_getitem(image_pair_preference_dir):  # noqa: F811
    dataset = ImagePairPreferenceDataset(str(image_pair_preference_dir))

    example = dataset[0]

    assert set(example.keys()) == {"id", "image_0", "image_1", "caption", "prefer_0", "prefer_1"}

    assert example["id"] == "0"

    assert isinstance(example["image_0"], PIL.Image.Image)
    assert example["image_0"].mode == "RGB"
    assert isinstance(example["image_1"], PIL.Image.Image)
    assert example["image_1"].mode == "RGB"

    assert example["prefer_0"]
    assert not example["prefer_1"]

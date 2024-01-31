import pytest
import torch

from invoke_training._shared.checkpoints.serialization import save_state_dict


@pytest.fixture(scope="session")
def sdv1_embedding_path(tmp_path_factory: pytest.TempPathFactory):
    """A fixture that writes a dummy SD v1 TI embedding to a temp dir and returns the embedding path.

    Note that the 'session' scope is used to share the same directory across all tests in a session. Refer to
    https://docs.pytest.org/en/7.4.x/how-to/tmp_path.html#the-tmp-path-factory-fixture for details on the use of
    tmp_path_factory.
    """
    tmp_dir = tmp_path_factory.mktemp("embeddings")

    embedding_state_dict = {"custom_token": torch.zeros((2, 768))}

    embedding_path = tmp_dir / "embedding.safetensors"
    save_state_dict(embedding_state_dict, embedding_path)

    return embedding_path


@pytest.fixture(scope="session")
def sdxl_embedding_path(tmp_path_factory: pytest.TempPathFactory):
    """A fixture that writes a dummy SDXL TI embedding to a temp dir and returns the embedding path.

    Note that the 'session' scope is used to share the same directory across all tests in a session. Refer to
    https://docs.pytest.org/en/7.4.x/how-to/tmp_path.html#the-tmp-path-factory-fixture for details on the use of
    tmp_path_factory.
    """
    tmp_dir = tmp_path_factory.mktemp("embeddings")

    embedding_state_dict = {
        "clip_l": torch.zeros((2, 768)),
        "clip_g": torch.zeros((2, 1280)),
    }

    embedding_path = tmp_dir / "embedding.safetensors"
    save_state_dict(embedding_state_dict, embedding_path)

    return embedding_path

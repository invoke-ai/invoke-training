import pytest

from invoke_training.lora.lora_training import run_lora_training


def test_run_lora_training():
    with pytest.raises(NotImplementedError):
        run_lora_training()

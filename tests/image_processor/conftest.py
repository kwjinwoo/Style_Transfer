import pytest

from style_transfer.transfer.config import TransferConfig


@pytest.fixture
def config():
    config_dict = {"normalize_mean": [2, 2, 2], "normalize_std": [0.5, 0.5, 0.5]}
    return TransferConfig(**config_dict)

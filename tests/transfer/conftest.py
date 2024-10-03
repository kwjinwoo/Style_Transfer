import pytest

from style_transfer.transfer.config import TransferConfig


@pytest.fixture
def config() -> TransferConfig:
    temp_config = {
        "num_steps": 2,
        "style_weight": 1,
        "content_weight": 1,
    }
    return TransferConfig(**temp_config)

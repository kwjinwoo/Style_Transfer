from style_transfer.transfer.config import TransferConfig
from style_transfer.utils import load_config


def test_load_config(transfer_config_path):
    transfer_config = load_config(transfer_config_path)

    assert isinstance(transfer_config, TransferConfig)

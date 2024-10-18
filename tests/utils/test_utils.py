import numpy as np
import torch
from PIL import Image

from style_transfer.transfer.config import TransferConfig
from style_transfer.utils import load_config, save_image


def test_load_config(transfer_config_path):
    transfer_config = load_config(transfer_config_path)

    assert isinstance(transfer_config, TransferConfig)


def test_save_image(tmp_path):
    temp_image = torch.clamp(torch.randn(1, 3, 10, 10), 0, 1)

    save_image(temp_image, tmp_path)

    temp_image = Image.open(tmp_path / "transfer_image.jpg")
    temp_image = np.array(temp_image)

    assert temp_image.shape == (10, 10, 3)

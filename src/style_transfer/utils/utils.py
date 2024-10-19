import json
import os
from typing import Dict

import torch
from torchvision import transforms

from style_transfer.transfer.config import TransferConfig


def load_json(path: str) -> Dict:
    """load json.

    Args:
        path (str): json config path.

    Returns:
        Dict: json dictionary.
    """
    with open(path, "r") as f:
        loaded_json = json.load(f)
    return loaded_json


def load_config(path: str) -> TransferConfig:
    """load Transfer config.

    Args:
        path (str): config paht.

    Returns:
        TransferConfig: Transfer config.
    """
    json_config = load_json(path)
    return TransferConfig(**json_config)


def save_image(image_tensor: torch.Tensor, save_dir: str) -> None:
    """save image from tensor.

    Args:
        image_tensor (torch.Tensor): image tensor to save.
        save_path (str): save directory.
    """
    image_tensor = image_tensor.cpu().clone().squeeze(0)

    pil_image = transforms.ToPILImage()(image_tensor)

    save_path = os.path.join(save_dir, "transfer_image.jpg")
    pil_image.save(save_path)


def init_dir(save_dir: str) -> None:
    """if save dir is not exist, make dir.

    Args:
        save_dir (str): dir path.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

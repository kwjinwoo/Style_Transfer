import json
from typing import Dict

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

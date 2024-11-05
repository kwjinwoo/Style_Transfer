import json

import pytest


@pytest.fixture
def transfer_config_path(tmp_path):
    config_path = tmp_path / "config.json"

    config = {
        "num_steps": 2,
        "style_weight": 1,
        "content_weight": 1,
        "normalize_mean": [1, 1, 1],
        "normalize_std": [2, 2, 2],
    }

    with open(config_path, "w") as f:
        json.dump(config, f)
    return config_path

import torch.nn as nn


def get_module(target: str, origin_module: nn.Module) -> nn.Module:
    attrs = target.split(".")

    module = origin_module
    for attr_name in attrs:
        module = getattr(module, attr_name)
    return module

import torch.nn as nn


def get_module(target: str, origin_module: nn.Module) -> nn.Module:
    """get module from node's target.

    Args:
        target (str): node's target. it is concatenated with `.`
        origin_module (nn.Module): origin model module.

    Returns:
        nn.Module: nn.Module of target.
    """
    attrs = target.split(".")

    module = origin_module
    for attr_name in attrs:
        module = getattr(module, attr_name)
    return module

import dataclasses

import torch


@dataclasses
class TransferConfig:
    num_steps: int = 300
    style_weight: float = 1000000.0
    content_weight: float = 1.0
    normalize_mean: torch.Tensor = torch.tensor([0.485, 0.456, 0.406])
    normalize_std: torch.Tensor = torch.tensor([0.229, 0.224, 0.225])

import torch
import torch.nn as nn
from torch.optim import LBFGS, Optimizer

from style_stransfer.transfer.config import TransferConfig


class Transfer:
    def __init__(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        feature_extarctor: nn.Module,
        transfer_config: TransferConfig,
    ) -> None:
        self.content_img = content_img
        self.style_img = style_img
        self.feature_extractor = feature_extarctor
        self.transfer_config = transfer_config

        self.optimizer = self.get_optimizer()
        self.nomalizer = self.get_normalizer()

    def get_optimizer(self) -> Optimizer:
        return LBFGS([self.content_img])

    def get_normalizer(self) -> nn.Module:
        pass

    def run() -> torch.Tensor:
        pass

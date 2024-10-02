from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import LBFGS, Optimizer

from style_stransfer.models import Normalizer
from style_stransfer.transfer.config import TransferConfig


class Transfer:
    def __init__(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        feature_extarctor: nn.Module,
        transfer_config: TransferConfig,
    ) -> None:
        """Style Transfer module.

        Args:
            content_img (torch.Tensor): content image.
            style_img (torch.Tensor): style image.
            feature_extarctor (nn.Module): feature extractor.
            transfer_config (TransferConfig): config.
        """
        self.content_img = content_img.requires_grad_(True)
        self.style_img = style_img.requires_grad_(False)
        self.feature_extractor = feature_extarctor.eval().requires_grad_(False)
        self.transfer_config = transfer_config

        self.optimizer = self.get_optimizer()
        self.nomalizer = self.get_normalizer().eval().requires_grad_(False)

    def get_optimizer(self) -> Optimizer:
        """get optimizer.

        Returns:
            Optimizer: optimizer.
        """
        return LBFGS([self.content_img])

    def get_normalizer(self) -> Normalizer:
        """get image normalizer.

        Returns:
            Normalizer: normalizer module.
        """
        return Normalizer(self.transfer_config.normalize_mean, self.transfer_config.normalize_std)

    def set_device(self) -> Tuple[torch.Tensor, torch.Tensor, nn.Module, nn.Module]:
        """set device.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, nn.Module, nn.Module]:
                content image, style image, feature extractor, normalizer.
        """
        device = self.transfer_config.device
        return (
            self.content_img.to(device=device),
            self.style_img.to(device=device),
            self.feature_extractor.to(device=device),
            self.nomalizer.to(device=device),
        )

    def run(self) -> torch.Tensor:
        """perfomrs style transfer.

        Returns:
            torch.Tensor: transfered image.
        """
        content_img, style_img, feature_extractor, normalizer = self.set_device()

        content_img = normalizer(content_img)
        style_img = normalizer(style_img)
        for _ in range(self.transfer_config.num_steps):
            content_features = feature_extractor(content_img)
            style_features = feature_extractor(style_img)

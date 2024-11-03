import torch
import torch.nn as nn
from torch.optim import Adam

from style_transfer.feature_extractor import Normalizer
from style_transfer.loss import ContentLoss, StyleLoss
from style_transfer.transfer.config import TransferConfig


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
        self.gen_img = content_img.clone()
        self.content_img = content_img
        self.style_img = style_img
        self.feature_extractor = feature_extarctor.eval()
        self.transfer_config = transfer_config

        self.nomalizer = self.get_normalizer().eval()

        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss()

        self.content_feature_idx = 3

    def get_optimizer(self, gen_img: torch.Tensor) -> Adam:
        """get optimizer.

        Args:
            gen_img (torch.Tensor): gen image to be optimized.
        Returns:
            Optimizer: optimizer.
        """
        return Adam([gen_img], lr=1)

    def get_normalizer(self) -> Normalizer:
        """get image normalizer.

        Returns:
            Normalizer: normalizer module.
        """
        return Normalizer(self.transfer_config.normalize_mean, self.transfer_config.normalize_std)

    def set_device(self) -> None:
        """set device."""
        device = torch.device(self.transfer_config.device)

        self.gen_img = self.gen_img.to(device=device)
        self.content_img = self.content_img.to(device=device)
        self.style_img = self.style_img.to(device=device)
        self.nomalizer = self.nomalizer.to(device=device)
        self.feature_extractor = self.feature_extractor.to(device=device)

    def de_normalize(self, img: torch.Tensor) -> torch.Tensor:
        """image de normalization.

        Args:
            img (torch.Tensor): input image.

        Returns:
            torch.Tensor: denormalized image.
        """
        mean = torch.tensor(self.transfer_config.normalize_mean).view(-1, 1, 1).to(device=img.device)
        std = torch.tensor(self.transfer_config.normalize_std).view(-1, 1, 1).to(device=img.device)
        return std * img + mean

    def run(self) -> torch.Tensor:
        """perfomrs style transfer.

        Returns:
            torch.Tensor: transfered image.
        """
        self.set_device()
        content_img = self.nomalizer(self.content_img)
        style_img = self.nomalizer(self.style_img)
        optimizer = self.get_optimizer(self.gen_img.requires_grad_(True))
        content_features = self.feature_extractor(content_img)
        style_features = self.feature_extractor(style_img)

        for step in range(self.transfer_config.num_steps):
            gen_img = self.nomalizer(self.gen_img)
            optimizer.zero_grad()

            gen_features = self.feature_extractor(gen_img)

            content_loss = self.content_loss(
                gen_features[self.content_feature_idx], content_features[self.content_feature_idx]
            )
            style_loss = self.style_loss(gen_features, style_features)

            total_loss = (
                self.transfer_config.content_weight * content_loss + self.transfer_config.style_weight * style_loss
            )
            total_loss.backward(retain_graph=True)

            optimizer.step()

            print(
                f"step_{step} total loss: {total_loss.item()} style loss: {style_loss.item()} "
                f"content loss: {content_loss.item()}"
            )

        return self.de_normalize(torch.clamp(self.gen_img, 0, 1))

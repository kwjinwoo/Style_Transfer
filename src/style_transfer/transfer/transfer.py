import torch
from torch.optim import Adam

from style_transfer.feature_extractor import FeatureExtractor
from style_transfer.loss import ContentLoss, StyleLoss
from style_transfer.transfer.config import TransferConfig


class Transfer:
    def __init__(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        gen_img: torch.Tensor,
        feature_extarctor: FeatureExtractor,
        transfer_config: TransferConfig,
    ) -> None:
        """Style Transfer module.

        Args:
            content_img (torch.Tensor): content image.
            style_img (torch.Tensor): style image.
            gen_img (torch.Tensor): generate image.
            feature_extarctor (FeatureExtractor): feature extractor.
            transfer_config (TransferConfig): config.
        """
        self.gen_img = gen_img
        self.content_img = content_img
        self.style_img = style_img
        self.feature_extractor = feature_extarctor
        self.transfer_config = transfer_config

        self.__set_transfer_attr()

        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss()

    def __set_transfer_attr(self) -> None:
        """set transfer config to class attributes."""
        self.num_steps = self.transfer_config.num_steps
        self.content_feature_idx = self.transfer_config.content_layer_num
        self.content_weight = self.transfer_config.content_weight
        self.style_weight = self.transfer_config.style_weight

    def get_optimizer(self, gen_img: torch.Tensor) -> Adam:
        """get optimizer.

        Args:
            gen_img (torch.Tensor): gen image to be optimized.
        Returns:
            Optimizer: optimizer.
        """
        return Adam([gen_img], lr=0.01)

    def set_device(self) -> None:
        """set device."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gen_img = self.gen_img.to(device=device)
        self.content_img = self.content_img.to(device=device)
        self.style_img = self.style_img.to(device=device)
        self.feature_extractor.extractor = self.feature_extractor.extractor.to(device=device)

    def run(self) -> torch.Tensor:
        """perfomrs style transfer.

        Returns:
            torch.Tensor: transfered image.
        """
        self.set_device()
        optimizer = self.get_optimizer(self.gen_img.requires_grad_(True))
        content_features = self.feature_extractor(self.content_img)
        style_features = self.feature_extractor(self.style_img)

        for step in range(self.num_steps):
            optimizer.zero_grad()

            gen_features = self.feature_extractor(self.gen_img)

            content_loss = self.content_loss(
                gen_features[self.content_feature_idx], content_features[self.content_feature_idx]
            )
            style_loss = self.style_loss(gen_features, style_features)

            total_loss = self.content_weight * content_loss + self.style_weight * style_loss
            total_loss.backward(retain_graph=True)

            optimizer.step()

            print(
                f"step_{step} total loss: {total_loss.item()} style loss: {style_loss.item()} "
                f"content loss: {content_loss.item()}"
            )

        return self.gen_img

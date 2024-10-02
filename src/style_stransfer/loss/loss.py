import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        """Stlye loss."""
        super().__init__(**kwargs)

    def get_gram_matrix(self, feature: torch.Tensor) -> torch.Tensor:
        """get gram matrix.

        Args:
            feature (torch.Tensor): input feature.

        Returns:
            torch.Tensor: gram matrix.
        """
        a, b, c, d = feature.size()
        feature = feature.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())
        return gram.div(a * b * c * d)

    def forward(self, content_feature: torch.Tensor, style_feature: torch.Tensor) -> torch.Tensor:
        """calcuate style loss.

        Args:
            content_feature (torch.Tensor): content feature.
            style_feature (torch.Tensor): style feature.

        Returns:
            torch.Tensor: style loss.
        """
        content_gram = self.get_gram_matrix(content_feature)
        style_gram = self.get_gram_matrix(style_feature)

        return F.mse_loss(content_gram, style_gram)

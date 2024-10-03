from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        """Stlye loss"""
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

    def forward(self, gen_feature: torch.Tensor, style_feature: torch.Tensor) -> torch.Tensor:
        """calcuate style loss.

        Args:
            gen_feature (torch.Tensor): generation image feature.
            style_feature (torch.Tensor): style feature.

        Returns:
            torch.Tensor: style loss.
        """
        gen_gram = self.get_gram_matrix(gen_feature)
        style_gram = self.get_gram_matrix(style_feature)

        return F.mse_loss(gen_gram, style_gram)


class ContentLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        """Content loss"""
        super().__init__(**kwargs)

    def forward(self, gen_features: List[torch.Tensor], content_features: List[torch.Tensor]) -> torch.Tensor:
        """calculate content loss.

        Args:
            gen_features (List[torch.Tensor]): generation image features.
            content_features (List[torch.Tensor]): content image features.

        Returns:
            torch.Tensor: content loss.
        """
        loss = 0
        for gen_feature, content_feature in zip(gen_features, content_features):
            loss += F.mse_loss(gen_feature, content_feature)
        return loss

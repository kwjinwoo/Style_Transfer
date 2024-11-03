import torch.nn as nn
from torchvision.models.vgg import VGG16_Weights, VGG19_Weights, vgg16, vgg19


class BaseModelMaker:
    """Base Model Factory class."""

    def __init__(self, base_model_name: str) -> None:
        self.base_model_name = base_model_name

        self.base_model_map = {"vgg19": vgg19, "vgg16": vgg16}
        self.base_weight_map = {vgg19: VGG19_Weights.IMAGENET1K_V1, vgg16: VGG16_Weights.IMAGENET1K_V1}

    def get_base_model(self) -> nn.Module:
        """get base model.

        Raises:
            ValueError: if base model name is not in base model map, raise ValueError.

        Returns:
            nn.Module: base model.
        """
        if self.base_model_name in self.base_model_map:
            base_model_class = self.base_model_map[self.base_model_name]
            base_model_weight = self.base_weight_map[base_model_class]
            return base_model_class(weights=base_model_weight).eval()
        else:
            raise ValueError(
                f"the base model name is not Supproted {self.base_model_name}."
                "Base model name Must be in {self.base_model_map.key()}"
            )

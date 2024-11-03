import pytest
import torch
from torchvision.models.vgg import VGG16_Weights, VGG19_Weights, vgg16, vgg19

from style_transfer.feature_extractor.base_model_maker import BaseModelMaker


@pytest.mark.parametrize(
    "name, answer",
    [("vgg19", vgg19(weights=VGG19_Weights.IMAGENET1K_V1)), ("vgg16", vgg16(weights=VGG16_Weights.IMAGENET1K_V1))],
)
def test_base_model_maker(name, answer):
    maker = BaseModelMaker(name)

    base_model = maker.get_base_model()

    for param, answer_param in zip(base_model.parameters(), answer.parameters()):
        assert torch.allclose(param, answer_param)

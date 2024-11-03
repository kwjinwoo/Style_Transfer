import pytest
import torch
import torch.nn as nn

from style_transfer.feature_extractor import get_feature_extractor
from style_transfer.feature_extractor.base_model_maker import BaseModelMaker
from style_transfer.feature_extractor.feature_extractor import FeatureExtractor


def get_answer_features(name, conv_nums, x):
    base_model = BaseModelMaker(name).get_base_model().features

    conv_idx = 1
    max_conv_idx = max(conv_nums)
    answer_features = []
    for i, layer in enumerate(base_model.children()):
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            if i in conv_nums:
                answer_features.append(x)
            conv_idx += 1

        if conv_idx >= max_conv_idx:
            break
    return answer_features


@pytest.mark.parametrize("name, conv_numbers", [("vgg19", [1, 3]), ("vgg16", [1, 2])])
def test_get_feature_extractor(name, conv_numbers):
    base_model = BaseModelMaker(name).get_base_model()
    extractor = FeatureExtractor(base_model, conv_numbers)

    temp_inp = torch.randn(1, 3, 224, 224)

    for param in extractor.extractor.parameters():
        assert param.requires_grad is False
    features = extractor(temp_inp)

    assert len(features) == len(conv_numbers)


@pytest.mark.parametrize("name, conv_nums", [("vgg19", [1, 3]), ("vgg16", [1, 2])])
def test_get_featurea_extractor(name, conv_nums):
    extractor = get_feature_extractor(name, conv_nums)

    temp_inp = torch.randn(1, 3, 224, 224)
    features = extractor(temp_inp)
    answer_features = get_answer_features(name, conv_nums, temp_inp)

    for feature, answer_feature in zip(features, answer_features):
        assert torch.allclose(feature, answer_feature)

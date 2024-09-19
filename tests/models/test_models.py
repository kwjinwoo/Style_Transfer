import torch

from style_stransfer.models import get_feature_extractor


def test_get_feature_extractor():
    feature_extractor = get_feature_extractor()
    temp_inp = torch.randn(1, 3, 224, 224)

    features = feature_extractor(temp_inp)
    feature_extractor.graph.print_tabular()
    assert len(features) == 5

import torch

from style_transfer.models import Normalizer, get_feature_extractor


def test_get_feature_extractor():
    feature_extractor = get_feature_extractor()
    temp_inp = torch.randn(1, 3, 224, 224)

    features = feature_extractor(temp_inp)
    feature_extractor.graph.print_tabular()
    assert len(features) == 5


def test_normalizer():
    normalizer = Normalizer(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))

    temp_img = torch.randn(1, 3, 10, 10)

    normalized_img = normalizer(temp_img)

    assert normalized_img.size() == torch.Size([1, 3, 10, 10])

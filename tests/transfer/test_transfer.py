import torch

from style_transfer.feature_extractor import get_feature_extractor
from style_transfer.transfer import Transfer


def test_transfer(config):
    temp_content = torch.randn(1, 3, 10, 10)
    temp_style = torch.randn(1, 3, 10, 10)
    temp_gen = temp_content.clone()
    feature_extractor = get_feature_extractor("vgg19", [1, 2, 3, 4, 5])

    transfer = Transfer(temp_content, temp_style, temp_gen, feature_extractor, config)

    gen_img = transfer.run()

    assert gen_img.size() == torch.Size([1, 3, 10, 10])

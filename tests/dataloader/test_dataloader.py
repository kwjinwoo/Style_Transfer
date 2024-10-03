import torch

from style_transfer.dataloader import get_content_style_img


def test_get_content_style_img() -> None:
    content_path = "img\\dancing.jpg"
    style_path = "img\\picasso.jpg"

    content_img, style_img = get_content_style_img(content_path, style_path)

    assert isinstance(content_img, torch.Tensor)
    assert isinstance(style_img, torch.Tensor)
    assert content_img.size() == torch.Size([1, 3, 512, 512])
    assert style_img.size() == torch.Size([1, 3, 512, 512])

import torch

from style_stransfer.loss import StyleLoss


def test_style_loss():
    style_loss = StyleLoss()

    temp_content = torch.randn(1, 3, 10, 10)
    temp_style = torch.randn(1, 3, 10, 10)

    content_gram = style_loss.get_gram_matrix(temp_content)
    style_gram = style_loss.get_gram_matrix(temp_style)

    out_loss = style_loss(temp_content, temp_style)

    assert content_gram.size() == torch.Size([3, 3])
    assert style_gram.size() == torch.Size([3, 3])
    assert out_loss.size() == torch.Size([])

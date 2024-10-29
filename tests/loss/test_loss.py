import torch

from style_transfer.loss import ContentLoss, StyleLoss


def test_style_loss():
    style_loss_fn = StyleLoss()

    num_features = 4

    temp_gen = []
    temp_style = []
    for _ in range(num_features):
        temp_gen.append(torch.randn(1, 3, 10, 10))
        temp_style.append(torch.randn(1, 3, 10, 10))

    style_loss = style_loss_fn(temp_gen, temp_style)
    assert style_loss.size() == torch.Size([])


def test_content_loss():
    content_loss_fn = ContentLoss()

    num_features = 4

    temp_gen = []
    temp_content = []
    for _ in range(num_features):
        temp_gen.append(torch.randn(1, 3, 10, 10))
        temp_content.append(torch.randn(1, 3, 10, 10))

    content_loss = content_loss_fn(torch.randn(1, 3, 10, 10), torch.randn(1, 3, 10, 10))
    assert content_loss.size() == torch.Size([])

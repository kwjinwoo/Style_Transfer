import torch

from style_stransfer.loss import ContentLoss, StyleLoss


def test_style_loss():
    style_loss_fn = StyleLoss()

    temp_gen = torch.randn(1, 3, 10, 10)
    temp_style = torch.randn(1, 3, 10, 10)

    content_gram = style_loss_fn.get_gram_matrix(temp_gen)
    style_gram = style_loss_fn.get_gram_matrix(temp_style)

    style_loss = style_loss_fn(temp_gen, temp_style)

    assert content_gram.size() == torch.Size([3, 3])
    assert style_gram.size() == torch.Size([3, 3])
    assert style_loss.size() == torch.Size([])


def test_content_loss():
    content_loss_fn = ContentLoss()

    num_features = 4

    temp_gen = []
    temp_content = []
    for _ in range(num_features):
        temp_gen.append(torch.randn(1, 3, 10, 10))
        temp_content.append(torch.randn(1, 3, 10, 10))

    content_loss = content_loss_fn(temp_gen, temp_content)

    assert content_loss.size() == torch.Size([])

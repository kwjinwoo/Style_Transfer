from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms


def get_content_style_img(
    content_img_path: str, style_img_path: str, img_size: int = 512
) -> Tuple[torch.Tensor, torch.Tensor]:
    """get content and style image tensors.

    Args:
        content_img_path (str): content image path.
        style_img_path (str): style image path
        img_size (int, optional): image size. Defaults to 512.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: content image tensor, style image tensor.
    """
    transformer = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

    content_img = Image.open(content_img_path)
    style_img = Image.open(style_img_path)

    return transformer(content_img).unsqueeze(0), transformer(style_img).unsqueeze(0)

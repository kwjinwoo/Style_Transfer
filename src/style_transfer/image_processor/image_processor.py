from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms

from style_transfer.transfer.config import TransferConfig


class ImageProcessor:
    """Image Processor.
    it loads images from image path, and normalize it. and copy from content image to make gen image.
    """

    def __init__(self, content_img_path: str, style_img_path: str, config: TransferConfig) -> None:
        self.content_img_path = content_img_path
        self.style_img_path = style_img_path
        self.config = config

        self.image_height = self.config.image_height
        self.image_width = self.config.image_width

        self.means = self.config.normalize_mean
        self.std = self.config.normalize_std

        self.pil_2_tensor = transforms.Compose(
            [
                transforms.Resize(size=(self.image_height, self.image_width)),
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.std),
            ]
        )

    def __image_load(self, path: str) -> torch.Tensor:
        """image load, resize, and normalize.

        Args:
            path (str): image path.

        Returns:
            torch.Tensor: resized, normalized image.
        """
        image = Image.open(path)
        return self.pil_2_tensor(image).unsqueeze(0)

    def get_train_images(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """get train images.
        it returns normalized content image, style image, and generated image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: content image, style image, generate image.
        """
        content_image = self.__image_load(self.content_img_path)
        style_image = self.__image_load(self.style_img_path)
        gen_image = content_image.clone()
        return content_image, style_image, gen_image

    def post_processing(self, image: torch.Tensor) -> torch.Tensor:
        """post processing.
        de-normalize input image.

        Args:
            image (torch.Tensor): input image.

        Returns:
            torch.Tensor: post processed image.
        """
        mean = torch.tensor(self.means).view(-1, 1, 1).to(image.device)
        std = torch.tensor(self.std).view(-1, 1, 1).to(image.device)
        return std * image + mean

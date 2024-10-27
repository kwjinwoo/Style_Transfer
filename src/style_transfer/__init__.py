import argparse

import torch

from style_transfer.dataloader import get_content_style_img
from style_transfer.models import get_feature_extractor
from style_transfer.transfer import Transfer
from style_transfer.utils import init_dir, load_config, save_image


def transfer_image(content_img_path: str, style_img_path: str, config_path: str) -> torch.Tensor:
    """image style transfer.

    Args:
        content_img_path (str): content image path.
        style_img_path (str): style image path.
        config_path (str): config path.

    Returns:
        torch.Tensor: transfered image.
    """
    content_img, style_img = get_content_style_img(content_img_path, style_img_path)
    feature_extractor = get_feature_extractor()
    transfer_config = load_config(config_path)

    transfer = Transfer(content_img, style_img, feature_extractor, transfer_config)
    return transfer.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Image style Transfer.")
    parser.add_argument("-c", "--content_img", action="store", type=str, required=True, help="content image path.")
    parser.add_argument("-s", "--style_img", action="store", type=str, required=True, help="style image path.")
    parser.add_argument("-f", "--config_path", action="store", type=str, required=True, help="config json paht.")
    parser.add_argument("-d", "--save_dir", action="store", type=str, required=True, help="save dir path.")

    args = parser.parse_args()

    content_image_path = args.content_img
    style_image_path = args.style_img
    config_path = args.config_path
    save_dir = args.save_dir

    init_dir(save_dir)

    transfered_image = transfer_image(content_image_path, style_image_path, config_path)
    save_image(transfered_image, save_dir)

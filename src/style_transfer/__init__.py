from style_transfer.dataloader import get_content_style_img
from style_transfer.models import get_feature_extractor
from style_transfer.transfer import Transfer
from style_transfer.utils import load_config


def transfer(content_img_path: str, style_img_path: str, config_path: str) -> None:
    content_img, style_img = get_content_style_img(content_img_path, style_img_path)
    feature_extractor = get_feature_extractor()
    transfer_config = load_config(config_path)

    transfer = Transfer(content_img, style_img, feature_extractor, transfer_config)
    tranfered_img = transfer.run()

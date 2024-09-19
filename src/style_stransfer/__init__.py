from style_stransfer.dataloader import get_content_style_img
from style_stransfer.models import get_feature_extractor


def transfer(content_img_path: str, style_img_path) -> None:
    content_img, style_img = get_content_style_img(content_img_path, style_img_path)
    feature_extractor = get_feature_extractor()

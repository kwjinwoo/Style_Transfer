from typing import List

from .base_model_maker import BaseModelMaker
from .feature_extractor import FeatureExtractor


def get_feature_extractor(base_model_name: str, conv_numbers: List[int]) -> FeatureExtractor:
    maker = BaseModelMaker(base_model_name)
    base_model = maker.get_base_model()

    feature_extractor = FeatureExtractor(base_model, conv_numbers)
    return feature_extractor


__all__ = ["get_feature_extractor"]
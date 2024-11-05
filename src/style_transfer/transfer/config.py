from dataclasses import dataclass, field
from typing import List


@dataclass
class TransferConfig:
    base_model_name: str = "vgg19"
    conv_layer_nums: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    content_layer_num: int = 3
    num_steps: int = 300
    style_weight: float = 1000000.0
    content_weight: float = 1.0
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    image_height: int = 512
    image_width: int = 512

from typing import List

import torch
import torch.nn as nn
from torch.fx import Graph, GraphModule, Node, symbolic_trace
from torchvision.models.vgg import VGG19_Weights, vgg19

from style_transfer.models.constants import CONV_IDX_LIST
from style_transfer.models.utils import get_module


def is_conv2d_module(node: Node, base_graph: GraphModule) -> bool:
    """check input node is conv2d nn.Module.

    Args:
        node (Node): fx Node.
        base_graph (GraphModule): traced fx graph.

    Returns:
        bool: if input node is conv2d nn.Module, return True. else False.
    """
    if node.op == "call_module":
        module = get_module(node.target, base_graph)
        if isinstance(module, nn.Conv2d):
            return True
    return False


def make_feature_extractor(base_model: nn.Module) -> GraphModule:
    """make feature extractor for style transfer.

    Args:
        base_model (nn.Module): pre-trained model.

    Returns:
        GraphModule: feature extractor.
    """
    base_graph = symbolic_trace(base_model)

    feature_extractor_graph = Graph()
    arg_dict = {}
    feature_extractor_outputs = []
    max_conv_idx = max(CONV_IDX_LIST)
    conv_idx = 0
    for node in base_graph.graph.nodes:
        new_node = feature_extractor_graph.node_copy(node, lambda x: arg_dict[x.target])
        arg_dict[new_node.target] = new_node

        if is_conv2d_module(new_node, base_graph):
            if conv_idx in CONV_IDX_LIST:
                feature_extractor_outputs.append(new_node)
            conv_idx += 1

        if conv_idx > max_conv_idx:
            break
    feature_extractor_graph.output(feature_extractor_outputs)
    return GraphModule(base_model, feature_extractor_graph)


def get_feature_extractor() -> GraphModule:
    """get feature extractor.

    Returns:
        GraphModule: feature extractor.
    """
    vgg = vgg19(weights=VGG19_Weights.DEFAULT).eval()
    return make_feature_extractor(vgg)


class Normalizer(nn.Module):
    def __init__(self, mean: List[float], std: List[float], **kwargs) -> None:
        """image normalizing module.

        Args:
            mean (List[float]): vgg19 mean values.
            std (List[float]): vgg19 std values.
        """
        super().__init__(**kwargs)

        mean = torch.tensor(mean)
        std = torch.tensor(std)

        self.mean = torch.nn.Parameter(mean.view(-1, 1, 1))
        self.std = torch.nn.Parameter(std.view(-1, 1, 1))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """noramlize input image.

        Args:
            img (torch.Tensor): input image.

        Returns:
            torch.Tensor: normalized image.
        """
        return (img - self.mean) / self.std

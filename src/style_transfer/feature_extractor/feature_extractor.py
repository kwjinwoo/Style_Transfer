from typing import List

import torch
import torch.nn as nn
from torch.fx import Graph, GraphModule, Node, symbolic_trace

from style_transfer.feature_extractor.utils import get_module


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


class FeatureExtractor:
    """feature extractor.
    it extract features of input conv numbers.
    """

    def __init__(self, base_model: nn.Module, conv_numbers: List[int]) -> None:
        self.base_model = self.__preprocess_base_model(base_model)
        self.conv_numbers = conv_numbers
        self.extractor = self.__make_feature_extractor()

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        """call feature extractor. it runs feature extractor's forward.

        Args:
            x (torch.Tensor): _description_

        Returns:
            List[torch.Tensor]: _description_
        """
        return self.forward(x)

    def __preprocess_base_model(self, base_model: nn.Module) -> nn.Module:
        """preprocessing base_model. change it eval mode, and freeze all params.

        Args:
            base_model (nn.Module): base network.

        Returns:
            nn.Module: preprocessed base model.
        """
        base_model = base_model.eval()

        for param in base_model.parameters():
            param.requires_grad_(False)
        return base_model

    def __make_feature_extractor(self) -> GraphModule:
        """make feature extractor for style transfer.

        Returns:
            GraphModule: feature extractor.
        """
        base_graph = symbolic_trace(self.base_model)

        feature_extractor_graph = Graph()
        arg_dict = {}
        feature_extractor_outputs = []
        max_conv_idx = max(self.conv_numbers)
        conv_idx = 1
        for node in base_graph.graph.nodes:
            new_node = feature_extractor_graph.node_copy(node, lambda x: arg_dict[x.target])
            arg_dict[new_node.target] = new_node

            if is_conv2d_module(new_node, base_graph):
                if conv_idx in self.conv_numbers:
                    feature_extractor_outputs.append(new_node)
                conv_idx += 1

            if conv_idx > max_conv_idx:
                break
        feature_extractor_graph.output(feature_extractor_outputs)
        return GraphModule(self.base_model, feature_extractor_graph).eval()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """return features.

        Args:
            x (torch.Tensor): input of featrure extractor.

        Returns:
            List[torch.Tensor]: extracted features.
        """
        return self.extractor(x)

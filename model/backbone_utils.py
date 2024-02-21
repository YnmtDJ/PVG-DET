from collections import OrderedDict
from typing import List, Optional, Callable

from torch import nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock


class BackboneWithFPN(nn.Module):
    """
    Feature Pyramid Network (FPN) backbone.
    It will return an OrderedDict[Tensor] with the same channels.
    """
    def __init__(
        self,
        backbone: nn.Module,
        in_channels_list: List[int],
        out_channels: int,
        returned_layers: Optional[List[str]] = None,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        """
        :param backbone: The backbone model used to extract different size features.
        :param in_channels_list: The number of channels of each feature map.
        :param out_channels: The number of output channels.
        :param returned_layers: The layers of the network to return. By default, all layers are returned.
        :param extra_blocks: If provided, it will be used to extract extra features.
        :param norm_layer: Module specifying the normalization layer to use. Default: None
        """
        super(BackboneWithFPN, self).__init__()
        self.out_channels = out_channels
        self.backbone = backbone
        self.returned_layers = returned_layers
        self.fpn = FeaturePyramidNetwork(in_channels_list, self.out_channels, extra_blocks, norm_layer)

    def forward(self, x):
        x = self.backbone(x)
        if self.returned_layers is not None:  # if the returned_layers is not None, only return the specified layers
            x = OrderedDict((key, x[key]) for key in self.returned_layers)
        x = self.fpn(x)
        return x

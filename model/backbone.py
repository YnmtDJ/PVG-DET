from typing import List, Optional

from torch import nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock


class BackboneWithFPN(nn.Module):
    """
    Feature Pyramid Network (FPN) backbone.
    It will return an OrderedDict[Tensor] with the same channels.
    """
    def __init__(self, backbone: nn.Module, in_channels_list: List[int], extra_blocks: Optional[ExtraFPNBlock] = None):
        """
        :param backbone: The backbone model used to extract different size features.
        :param in_channels_list: The number of channels of each feature map.
        :param extra_blocks: If provided, it will be used to extract extra features.
        """
        super(BackboneWithFPN, self).__init__()
        self.out_channels = 256
        self.backbone = backbone
        self.fpn = FeaturePyramidNetwork(in_channels_list, self.out_channels, extra_blocks)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        return x

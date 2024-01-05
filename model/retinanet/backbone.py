from torch import nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from model.vig.vig import PyramidViG


class PyramidBackbone(nn.Module):
    """
    Feature Pyramid Network (FPN) backbone.
    It will return an OrderedDict[Tensor] with the same channels.
    """
    def __init__(self):
        super(PyramidBackbone, self).__init__()
        channels = [80, 160, 400, 640]
        blocks = [2, 2, 6, 2]
        self.out_channels = 80
        self.backbone = PyramidViG(blocks, channels)
        self.fpn = FeaturePyramidNetwork([160, 400, 640], self.out_channels,
                                         LastLevelP6P7(self.out_channels, self.out_channels))

    def forward(self, x):
        x = self.backbone(x, return_intermediate=True)
        x = self.fpn(x)
        return x

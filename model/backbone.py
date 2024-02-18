from functools import partial

from torch import nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from model.pvt.pvt_v2 import PyramidVisionTransformerV2
from model.vig.vig import PyramidViG


class PyramidBackbone(nn.Module):
    """
    Feature Pyramid Network (FPN) backbone.
    It will return an OrderedDict[Tensor] with the same channels.
    """
    def __init__(self):
        super(PyramidBackbone, self).__init__()
        channels = [64, 128, 256, 512]
        blocks = [2, 2, 6, 2]
        sr_ratios = [4, 2, 1, 1]
        self.out_channels = 256
        # self.backbone = PyramidViG(blocks, channels, sr_ratios)
        self.backbone = PyramidVisionTransformerV2(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
                                                   mlp_ratios=[8, 8, 4, 4],
                                                   qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                                   depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
                                                   drop_rate=0.0, drop_path_rate=0.1, linear=True)
        self.fpn = FeaturePyramidNetwork([128, 320, 512], self.out_channels,
                                         LastLevelP6P7(self.out_channels, self.out_channels))

    def forward(self, x):
        x = self.backbone(x, return_intermediate=True)
        x = self.fpn(x)
        return x

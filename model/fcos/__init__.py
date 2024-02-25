from functools import partial

import torch
from torch import nn
from torchvision.models import resnet50, resnet18
from torchvision.models.detection import FCOS
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from ..backbone_utils import BackboneWithFPN
from ..pvt.pvt_v2 import pvt_v2_b2_li, pvt_v2_b1_li
from ..vig.vig import pvg_s, pvg_t


def build_fcos(opts):
    """
    Build the FCOS model.
    :param opts: The options.
    :return: model
    """
    if opts.backbone.startswith('pvg'):
        if opts.backbone == 'pvg_s':
            backbone = pvg_s(opts.k, opts.gcn, opts.drop_prob)
        elif opts.backbone == 'pvg_t':
            backbone = pvg_t(opts.k, opts.gcn, opts.drop_prob)
        else:
            raise ValueError("Unknown backbone.")
        backbone = BackboneWithFPN(
            backbone, backbone.out_channels_list[1:], 256, ["1", "2", "3"], LastLevelP6P7(256, 256)
        )
    elif opts.backbone.startswith('resnet'):
        if opts.backbone == 'resnet50':
            backbone = resnet50(norm_layer=partial(nn.GroupNorm, 32))
        elif opts.backbone == 'resnet18':
            backbone = resnet18(norm_layer=partial(nn.GroupNorm, 32))
        else:
            raise ValueError("Unknown backbone.")
        backbone = _resnet_fpn_extractor(backbone, 5, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256))
    elif opts.backbone.startswith('pvt'):
        if opts.backbone == 'pvt_v2_b2_li':
            backbone = pvt_v2_b2_li()
        elif opts.backbone == 'pvt_v2_b1_li':
            backbone = pvt_v2_b1_li()
        else:
            raise ValueError("Unknown backbone.")
        backbone = BackboneWithFPN(backbone, [128, 256, 512], 256, ["1", "2", "3"], LastLevelP6P7(256, 256))
    else:
        raise ValueError("Unknown backbone.")

    device = torch.device(opts.device)
    model = FCOS(backbone, opts.num_classes, opts.min_size, opts.max_size).to(device)
    return model

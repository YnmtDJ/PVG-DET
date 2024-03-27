from functools import partial

import torch
from torch import nn
from torchvision.models import resnet50, resnet18
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from model.backbone_utils import BackboneWithFPN
from model.faster_rcnn import build_fasterrcnn
from model.fcos import build_fcos
from model.pvt.pvt_v2 import pvt_v2_b2_li, pvt_v2_b1
from model.retinanet import build_retinanet
from model.vig.vig import pvg_s, pvg_t


def build(opts):
    """
    Build the model.
    :param opts: The options.
    :return: model
    """
    backbone = build_backbone(opts)
    if opts.baseline == 'retinanet':
        model = build_retinanet(backbone, opts.num_classes, opts.min_size, opts.max_size)
    elif opts.baseline == 'fcos':
        model = build_fcos(backbone, opts.num_classes, opts.min_size, opts.max_size)
    elif opts.baseline == 'fasterrcnn':
        model = build_fasterrcnn(backbone, opts.num_classes, opts.min_size, opts.max_size)
    else:
        raise ValueError("Unknown baseline.")

    device = torch.device(opts.device)
    return model.to(device)


def build_backbone(opts):
    """
    Build the backbone.
    :param opts: The options.
    :return: backbone
    """
    if opts.backbone.startswith('pvg'):
        if opts.backbone == 'pvg_s':
            backbone = pvg_s(opts.k, opts.gcn, opts.drop_prob)
        elif opts.backbone == 'pvg_t':
            backbone = pvg_t(opts.k, opts.gcn, opts.drop_prob)
        else:
            raise ValueError("Unknown backbone.")
        backbone = BackboneWithFPN(
            backbone, backbone.out_channels_list, 256, ["0", "1", "2", "3"],
            LastLevelMaxPool(), partial(nn.GroupNorm, 32)
        )
    elif opts.backbone.startswith('resnet'):
        if opts.backbone == 'resnet50':
            backbone = resnet50(norm_layer=partial(nn.GroupNorm, 32))
        elif opts.backbone == 'resnet18':
            backbone = resnet18(norm_layer=partial(nn.GroupNorm, 32))
        else:
            raise ValueError("Unknown backbone.")
        backbone = _resnet_fpn_extractor(backbone, 5, norm_layer=partial(nn.GroupNorm, 32))
    elif opts.backbone.startswith('pvt'):
        if opts.backbone == 'pvt_v2_b2_li':
            backbone = pvt_v2_b2_li()
        elif opts.backbone == 'pvt_v2_b1':
            backbone = pvt_v2_b1()
        else:
            raise ValueError("Unknown backbone.")
        backbone = BackboneWithFPN(
            backbone, [64, 128, 256, 512], 256, ["0", "1", "2", "3"],
            LastLevelMaxPool(), partial(nn.GroupNorm, 32)
        )
    else:
        raise ValueError("Unknown backbone.")

    return backbone

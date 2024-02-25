from functools import partial

import torch
from torch import nn
from torchvision.models import resnet50, resnet18
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from ..backbone_utils import BackboneWithFPN
from ..pvt.pvt_v2 import pvt_v2_b2_li, pvt_v2_b1_li
from ..vig.vig import pvg_s, pvg_t


def build_retinanet(opts):
    """
    Build the RetinaNet model.
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

    # TODO: different dataset design different anchor
    if opts.dataset_name == 'COCO':
        anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    elif opts.dataset_name == 'VisDrone':
        anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [8, 16, 32, 64, 128])
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    else:
        anchor_sizes = None
        aspect_ratios = None
        raise ValueError("Unknown dataset name.")
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    head = RetinaNetHead(
        backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        opts.num_classes,
        norm_layer=partial(nn.GroupNorm, 32),
    )
    head.regression_head._loss_type = "giou"

    device = torch.device(opts.device)
    model = RetinaNet(
        backbone, opts.num_classes, opts.min_size, opts.max_size,
        anchor_generator=anchor_generator, head=head
    ).to(device)
    return model

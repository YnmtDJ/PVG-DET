from functools import partial

import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from model.backbone_utils import BackboneWithFPN
from model.vig.vig import pvg_s


def build_fasterrcnn(opts):
    """
    Build the FasterRCNN model.
    :param opts: The options.
    :return: model
    """
    if opts.backbone == 'pvg_s':
        backbone = pvg_s(opts.k, opts.gcn, opts.drop_prob)
        backbone = BackboneWithFPN(
            backbone, backbone.out_channels_list, 256, ["0", "1", "2", "3"],
            LastLevelMaxPool(), partial(nn.GroupNorm, 32)
        )
    elif opts.backbone == 'resnet50':
        backbone = resnet50()
        backbone = _resnet_fpn_extractor(backbone, 5, norm_layer=partial(nn.GroupNorm, 32))
    else:
        raise ValueError("Unknown backbone.")

    # TODO: different dataset design different anchor
    if opts.dataset_name == 'COCO':
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    elif opts.dataset_name == 'VisDrone':
        anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    else:
        anchor_sizes = None
        aspect_ratios = None
        raise ValueError("Unknown dataset name.")
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=partial(nn.GroupNorm, 32)
    )

    device = torch.device(opts.device)
    model = FasterRCNN(
        backbone, opts.num_classes, opts.min_size, opts.max_size,
        rpn_anchor_generator=rpn_anchor_generator, rpn_head=rpn_head, box_head=box_head,
    ).to(device)
    return model

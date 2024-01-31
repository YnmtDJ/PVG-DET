from functools import partial

import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from ..backbone import PyramidBackbone

def build_retinanet(opts):
    """
    Build the RetinaNet model.
    :param opts: The options.
    :return: model
    """
    device = torch.device(opts.device)
    if opts.dataset_name == "COCO":
        num_classes = 91  # because the coco dataset max label id is 90, so we set the num_classes to 91
    elif opts.dataset_name == "VisDrone":
        num_classes = 12  # because the visdrone dataset max label id is 11, so we set the num_classes to 12
    elif opts.dataset_name == "ImageNet":
        raise NotImplementedError("ImageNet dataset is not implemented yet.")
    else:
        num_classes = 20  # default num_classes

    # backbone = PyramidBackbone()
    backbone = resnet50()
    backbone = _resnet_fpn_extractor(
        backbone, 5, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(2048, 256)
    )
    anchor_sizes = tuple([(6, 7, 8, 9, 10), (12, 14, 16, 20, 22), (24, 28, 32, 36, 40), (48, 56, 64, 72, 80),
                          (96, 112, 128, 144, 160)])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    head = RetinaNetHead(
        backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        num_classes,
        norm_layer=partial(nn.GroupNorm, 32),
    )
    head.regression_head._loss_type = "giou"
    # TODO: image resolution
    model = RetinaNet(backbone, num_classes, [512, 544, 576, 608, 640, 672, 704], 1072,
                      anchor_generator=anchor_generator, head=head,
                      score_thresh=0.05, fg_iou_thresh=0.35, bg_iou_thresh=0.25).to(device)
    return model
from functools import partial

import torch
from torch import nn
from torchvision.models import resnet50
from .fcos import FCOS
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import LastLevelP6P7


def build_fcos(opts):
    """
    Build the FCOS model.
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
    backbone = resnet50(norm_layer=partial(nn.GroupNorm, 32))
    backbone = _resnet_fpn_extractor(
        backbone, 5, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
    )
    # TODO: image resolution  [512, 544, 576, 608, 640, 672, 704]
    model = FCOS(backbone, num_classes, 800, 1333, score_thresh=0.2).to(device)
    return model

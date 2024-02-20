from functools import partial

import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models.detection import FCOS
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from model.backbone_utils import BackboneWithFPN
from model.retinanet import build_retinanet
from model.vig.vig import pvg_s


def build(opts):
    """
    Build the model and criterion.
    :param opts: The options.
    :return: model, criterion
    """
    if opts.backbone == 'pvg_s':
        backbone = pvg_s(opts.k, opts.gcn, opts.drop_prob)
        backbone = BackboneWithFPN(backbone, [128, 256, 512], LastLevelP6P7(256, 256))
    elif opts.backbone == 'resnet50':
        backbone = resnet50(norm_layer=partial(nn.GroupNorm, 32))
        backbone = _resnet_fpn_extractor(
            backbone, 5, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
        )
    else:
        raise ValueError("Unknown backbone.")

    device = torch.device(opts.device)
    model = FCOS(backbone, opts.num_classes, opts.min_size, opts.max_size).to(device)
    return model

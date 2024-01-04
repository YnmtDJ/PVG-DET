import torch
from torchvision.models import resnet50
from torchvision.models.detection import RetinaNet, retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from model.criterion import SetCriterion
from model.de_gcn import DeGCN
from model.retinanet.retinanet import PyramidBackbone


def build(opts):
    """
    Build the model and criterion.
    :param opts: The options.
    :return: model, criterion
    """
    device = torch.device(opts.device)
    if opts.dataset_name == "COCO":
        num_classes = 91  # because the coco dataset max label id is 90, so we set the num_classes to 91
    elif opts.dataset_name == "ImageNet":
        raise NotImplementedError("ImageNet dataset is not implemented yet.")
    else:
        num_classes = 20  # default num_classes

    #  TODO: try different models
    backbone = PyramidBackbone()
    # backbone = resnet50()
    # backbone = _resnet_fpn_extractor(
    #     backbone, 5, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(2048, 256)
    # )
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [16, 32, 64, 128, 256])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    model = RetinaNet(backbone, num_classes, 224, 400, anchor_generator=anchor_generator).to(device)  # model = DeGCN(num_classes).to(device)
    criterion = SetCriterion(num_classes).to(device)

    return model, criterion

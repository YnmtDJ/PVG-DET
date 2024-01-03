import torch
from torchvision.models.detection import RetinaNet, retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_Weights

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
    model = RetinaNet(backbone, num_classes, 224, 352).to(device)  # model = DeGCN(num_classes).to(device)
    model = retinanet_resnet50_fpn_v2(min_size=224, max_size=352).to(device)
    criterion = SetCriterion(num_classes).to(device)

    return model, criterion

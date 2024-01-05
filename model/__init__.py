import torch

from model.criterion import SetCriterion
from model.de_gcn import DeGCN
from model.retinanet import build_retinanet


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
    model = build_retinanet(opts)  # DeGCN(num_classes).to(device)
    criterion = SetCriterion(num_classes).to(device)

    return model, criterion

import torch

from model.fcos import build_fcos
from model.criterion import SetCriterion
from model.de_gcn import DeGCN
from model.retinanet import build_retinanet


def build(opts):
    """
    Build the model and criterion.
    :param opts: The options.
    :return: model, criterion
    """
    #  TODO: try different models
    if opts.baseline == 'RetinaNet':
        model = build_retinanet(opts)
    elif opts.baseline == 'FCOS':
        model = build_fcos(opts)
    else:
        raise ValueError("Unknown baseline.")
    criterion = SetCriterion(opts.num_classes).to(torch.device(opts.device))
    return model, criterion

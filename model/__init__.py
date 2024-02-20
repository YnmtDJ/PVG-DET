from model.backbone_utils import BackboneWithFPN
from model.fcos import build_fcos
from model.retinanet import build_retinanet
from model.vig.vig import pvg_s


def build(opts):
    """
    Build the model.
    :param opts: The options.
    :return: model
    """
    if opts.baseline == 'retinanet':
        model = build_retinanet(opts)
    elif opts.baseline == 'fcos':
        model = build_fcos(opts)
    else:
        raise ValueError("Unknown baseline.")

    return model

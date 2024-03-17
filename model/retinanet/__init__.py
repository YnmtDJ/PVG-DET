from functools import partial

from torch import nn
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetHead


def build_retinanet(backbone, num_classes, min_size, max_size):
    """
    Build the RetinaNet model.
    :param backbone: The network used to compute the features for the model.
        It should contain an out_channels attribute, which indicates the number of output
        channels that each feature map has (and it should be the same for all feature maps).
        The backbone should return a single Tensor or an OrderedDict[Tensor].
    :param num_classes: The number of classes.
    :param min_size: Minimum size of the image to be rescaled before feeding it to the backbone
    :param max_size: Maximum size of the image to be rescaled before feeding it to the backbone
    :return: model
    """
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    head = RetinaNetHead(
        backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        num_classes,
        norm_layer=partial(nn.GroupNorm, 32),
    )
    head.regression_head._loss_type = "giou"

    return RetinaNet(backbone, num_classes, min_size, max_size, anchor_generator=anchor_generator, head=head)

from functools import partial

from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead
from torchvision.models.detection.rpn import RPNHead


def build_fasterrcnn(backbone, num_classes, min_size, max_size):
    """
    Build the FasterRCNN model.
    :param backbone: The network used to compute the features for the model.
        It should contain an out_channels attribute, which indicates the number of output
        channels that each feature map has (and it should be the same for all feature maps).
        The backbone should return a single Tensor or an OrderedDict[Tensor].
    :param num_classes: The number of classes.
    :param min_size: Minimum size of the image to be rescaled before feeding it to the backbone
    :param max_size: Maximum size of the image to be rescaled before feeding it to the backbone
    :return: model
    """
    anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=partial(nn.GroupNorm, 32)
    )

    return FasterRCNN(
        backbone, num_classes, min_size, max_size,
        rpn_anchor_generator=rpn_anchor_generator, rpn_head=rpn_head, box_head=box_head,
    )

from torchvision.models.detection import FCOS
from torchvision.models.detection.anchor_utils import AnchorGenerator


def build_fcos(backbone, num_classes, min_size, max_size):
    """
    Build the FCOS model.
     :param backbone: The network used to compute the features for the model.
        It should contain an out_channels attribute, which indicates the number of output
        channels that each feature map has (and it should be the same for all feature maps).
        The backbone should return a single Tensor or an OrderedDict[Tensor].
    :param num_classes: The number of classes.
    :param min_size: Minimum size of the image to be rescaled before feeding it to the backbone
    :param max_size: Maximum size of the image to be rescaled before feeding it to the backbone
    :return: model
    """
    # For FCOS, only set one anchor for per position of each level, the width and height equal to the stride of feature
    # map, and set aspect ratio = 1.0, so the center of anchor is equivalent to the point in FCOS paper.
    anchor_sizes = ((4,), (8,), (16,), (32,), (64,))  # equal to strides of multi-level feature map
    aspect_ratios = ((1.0,),) * len(anchor_sizes)  # set only one anchor
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    return FCOS(backbone, num_classes, min_size, max_size, anchor_generator=anchor_generator)

import torch
from torchvision.transforms import v2


def create_transform():
    """
    Create the transform for the training and validation datasets.
    :return: transform_train, transform_val
    """
    normalize = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_train = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.RandomIoUCrop(),
        v2.RandomResize(224, 400, antialias=True),
        v2.SanitizeBoundingBoxes(),
        normalize
    ])

    transform_val = v2.Compose([
        v2.ToImage(),
        normalize
    ])

    return transform_train, transform_val

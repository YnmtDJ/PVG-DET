import torch
from torchvision.transforms import v2


def create_transform():
    """
    Create the transform for the training and validation datasets.
    :return: transform_train, transform_val
    """
    transform_train = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.RandomIoUCrop(min_scale=0.7),
        v2.SanitizeBoundingBoxes(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    transform_val = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    return transform_train, transform_val

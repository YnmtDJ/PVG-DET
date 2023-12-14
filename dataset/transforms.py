"""
Because the coco dataset has image which has no object,
so we use the `torchvision.transforms.v2` to define our transform which can handle the image with no object.
"""
import torch
import torchvision.transforms.v2 as v2


class TrainTransform(object):
    """
    The transform for the training set.
    """
    def __init__(self):
        normalize = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.transforms_box = v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(),
            v2.RandomIoUCrop(),
            v2.RandomResize(224, 400, antialias=True),
            v2.ConvertBoundingBoxFormat("CXCYWH"),
            v2.SanitizeBoundingBoxes(),
            normalize
        ])

        self.transforms_no_box = v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(),
            v2.RandomResize(224, 400, antialias=True),
            normalize
        ])

    def __call__(self, image, target):
        if 'boxes' in target:
            image, target = self.transforms_box(image, target)
        else:
            image, target = self.transforms_no_box(image, target)
        return image, target


class ValTransform(object):
    """
    The transform for the validation set.
    """
    def __init__(self):
        normalize = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.transforms_box = v2.Compose([
            v2.ToImage(),
            v2.ConvertBoundingBoxFormat("CXCYWH"),
            normalize
        ])

        self.transforms_no_box = v2.Compose([
            v2.ToImage(),
            normalize
        ])

    def __call__(self, image, target):
        if "boxes" in target:
            image, target = self.transforms_box(image, target)
        else:
            image, target = self.transforms_no_box(image, target)
        return image, target

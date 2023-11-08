import os
from enum import Enum

import torchvision.datasets


def create_dataset(opts):

    if opts.datasetName == "COCO":
        dataset = torchvision.datasets.CocoDetection(root=os.path.join(opts.datasetRoot, "COCO", "2017"),
                                                     annFile=os.path.join(opts.datasetRoot, "COCO", "annotations", "2017.json"))
    elif opts.datasetName == "ImageNet":
        dataset = torchvision.datasets.ImageNet(root=os.path.join(opts.datasetRoot, "ImageNet"))
    else:
        raise RuntimeError("this dataset does not exist")

    return dataset


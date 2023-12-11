import os

import cv2
import torch
import torchvision.datasets
from torchvision.transforms import v2


def create_coco_dataset(data_root: str, image_set: str):
    """
    Create the coco dataset.
    :param data_root: The root directory of coco dataset.
    :param image_set: The image set, e.g. train, val.
    :return: The torch.utils.data.Dataset().
    """
    normalize = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == "train":
        transforms = v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(),
            v2.RandomIoUCrop(),
            v2.RandomResize(224, 400, antialias=True),
            v2.ConvertBoundingBoxFormat("CXCYWH"),
            v2.SanitizeBoundingBoxes(),
            normalize
        ])
        root = os.path.join(data_root, "images/train2017")
        annFile = os.path.join(data_root, "annotations/instances_train2017.json")
    elif image_set == "val":
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ConvertBoundingBoxFormat("CXCYWH"),
            normalize
        ])
        root = os.path.join(data_root, "images/val2017")
        annFile = os.path.join(data_root, "annotations/instances_val2017.json")
    else:
        raise ValueError("The image_set must be train or val.")

    dataset = torchvision.datasets.CocoDetection(root=root, annFile=annFile, transforms=transforms)
    return torchvision.datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=["image_id", "boxes", "labels"])


if __name__ == "__main__":
    # demo for the create_coco_dataset()
    test_dataset = create_coco_dataset("./COCO/", "val")
    image, target = test_dataset[0]
    image = image.permute(1, 2, 0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for i in range(len(target['boxes'])):
        bbox = target['boxes'][i]
        label = target['labels'][i]
        print(label)
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255))

    cv2.imshow("demo", image)
    cv2.waitKey()


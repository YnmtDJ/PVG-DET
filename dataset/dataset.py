import os

import cv2
import torchvision.datasets
from torchvision.transforms import v2


def create_coco_dataset(root: str, annFile: str, augment: bool):
    """
    Create the coco dataset.
    :param root: Root directory where images are downloaded to.
    :param annFile: Path to json annotation file.
    :param augment: Whether to perform data augmentation.
    :return: The torch.utils.data.Dataset().
    """
    normalize = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
    ])

    if augment:
        size = (224, 224)  # the size of output image
        ratio = size[0]/size[1]  # the aspect ratio of the crop
        transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(ratio, ratio)),
                normalize
        ])
    else:
        transforms = normalize

    return torchvision.datasets.CocoDetection(root=root, annFile=annFile, transforms=torchvision.transforms.ToTensor())


if __name__ == "__main__":
    dataset = create_coco_dataset(root="./COCO/images/val2017", annFile="./COCO/annotations/instances_val2017.json", augment=True)
    image, target = dataset[0]
    image = image.permute(1, 2, 0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bbox = target[0]['bbox']

    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 0, 255), thickness=2)

    cv2.imshow("test", image)
    cv2.waitKey()


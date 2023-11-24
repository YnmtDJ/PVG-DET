import cv2
import torch
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
    normalize = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if augment:
        transforms = v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(),
            v2.RandomIoUCrop(),
            v2.RandomResize(224, 400, antialias=True),
            v2.SanitizeBoundingBoxes(),
            normalize
        ])
    else:
        transforms = v2.Compose([
            v2.ToImage(),
            normalize
        ])

    dataset = torchvision.datasets.CocoDetection(root=root, annFile=annFile, transforms=transforms)
    return torchvision.datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=["boxes", "labels"])


if __name__ == "__main__":
    # demo for the create_coco_dataset()
    test_dataset = create_coco_dataset(root="./COCO/images/val2017", annFile="./COCO/annotations/instances_val2017.json", augment=True)
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


import os

import cv2
import torch
import torchvision
import torchvision.transforms.v2.functional as F
from torchvision import tv_tensors

from dataset.transforms import create_transform
from util.misc import list_of_dicts_to_dict_of_lists


def create_dataset(dataset_root: str, dataset_name: str):
    """
    Create training and validation datasets.
    :param dataset_root: root directory of all datasets
    :param dataset_name: select the dataset to use
    :return: dataset_train, dataset_val
    """
    if dataset_name == "COCO":
        dataset_train = create_coco_dataset(os.path.join(dataset_root, dataset_name), "train")
        dataset_val = create_coco_dataset(os.path.join(dataset_root, dataset_name), "val")
    elif dataset_name == "ImageNet":
        raise NotImplementedError("ImageNet dataset is not implemented yet.")
    else:
        raise ValueError("Unknown dataset name.")

    return dataset_train, dataset_val


def create_coco_dataset(data_root: str, image_set: str):
    """
    Create the coco dataset.
    :param data_root: The root directory of coco dataset.
    :param image_set: The image set, e.g. train, val.
    :return: The torch.utils.data.Dataset().
    """
    transform_train, transform_val = create_transform()

    if image_set == "train":
        transforms = transform_train
        root = os.path.join(data_root, "images/train2017")
        annFile = os.path.join(data_root, "annotations/instances_train2017.json")
    elif image_set == "val":
        transforms = transform_val
        root = os.path.join(data_root, "images/val2017")
        annFile = os.path.join(data_root, "annotations/instances_val2017.json")
    else:
        raise ValueError("The image_set must be train or val.")

    return CocoDetection(root=root, annFile=annFile, transforms=transforms)


class CocoDetection(torchvision.datasets.CocoDetection):
    """
    Referring to the pytorch CocoDetection implementation, the difference is that
    when there are no objects in the image, empty boxes and labels are still returned.
    """
    def __init__(self, root, annFile, transforms=None):
        super(CocoDetection, self).__init__(root, annFile)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.wrap_dataset_for_transforms_v2(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

    def wrap_dataset_for_transforms_v2(self, image, target):
        canvas_size = tuple(F.get_size(image))

        batched_target = list_of_dicts_to_dict_of_lists(target['annotations'])
        target = {"image_id": target["image_id"]}

        if "bbox" in batched_target:
            target["boxes"] = F.convert_bounding_box_format(
                tv_tensors.BoundingBoxes(
                    batched_target["bbox"],
                    format=tv_tensors.BoundingBoxFormat.XYWH,
                    canvas_size=canvas_size,
                ),
                new_format=tv_tensors.BoundingBoxFormat.CXCYWH,
            )
        else:
            target["boxes"] = F.convert_bounding_box_format(
                tv_tensors.BoundingBoxes(
                    torch.empty((0, 4)),
                    format=tv_tensors.BoundingBoxFormat.XYWH,
                    canvas_size=canvas_size,
                ),
                new_format=tv_tensors.BoundingBoxFormat.CXCYWH,
            )

        if "category_id" in batched_target:
            target["labels"] = torch.tensor(batched_target["category_id"])
        else:
            target["labels"] = torch.empty(0, dtype=torch.int64)

        return image, target


if __name__ == "__main__":
    # demo for the create_dataset()
    dataset_train, dataset_val = create_dataset("./", "COCO")
    image, target = dataset_train[0]
    image = image.permute(1, 2, 0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for i in range(len(target['boxes'])):
        bbox = target['boxes'][i]
        label = target['labels'][i]
        center_x = int(bbox[0])
        center_y = int(bbox[1])
        width = int(bbox[2])
        height = int(bbox[3])
        print(label)
        cv2.rectangle(image, (center_x-width//2, center_y-height//2), (center_x+width//2, center_y+height//2),
                      (0, 0, 255))

    cv2.imshow("demo", image)
    cv2.waitKey()


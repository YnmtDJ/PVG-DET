import os
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image
import cv2
import torch
from torchvision import tv_tensors
import torchvision.transforms.v2.functional as F
from torchvision.transforms import v2

from dataset.transforms import TrainTransform, ValTransform
from util.misc import list_of_dicts_to_dict_of_lists


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, root, annFile, transforms):
        super(CocoDetection, self).__init__(root, annFile)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # TODO: use wrap_dataset_for_transforms_v2() instead of wrap_dataset_for_transforms()
        return img, target

    def wrap_dataset_for_transforms_v2(self, image, target):
        canvas_size = tuple(F.get_size(image))

        batched_target = list_of_dicts_to_dict_of_lists(target)
        target = {}

        target["image_id"] = image_id

        if "boxes" in batched_target:
            target["boxes"] = F.convert_bounding_box_format(
                tv_tensors.BoundingBoxes(
                    batched_target["bbox"],
                    format=tv_tensors.BoundingBoxFormat.XYWH,
                    canvas_size=canvas_size,
                ),
                new_format=tv_tensors.BoundingBoxFormat.XYXY,
            )
        else:
            target["boxes"] = F.convert_bounding_box_format(
                tv_tensors.BoundingBoxes(
                    torch.empty((0, 4)),
                    format=tv_tensors.BoundingBoxFormat.XYWH,
                    canvas_size=canvas_size,
                ),
                new_format=tv_tensors.BoundingBoxFormat.XYXY,
            )

        if "labels" in batched_target:
            target["labels"] = torch.tensor(batched_target["category_id"])
        else:
            target["labels"] = torch.empty(0, dtype=torch.int64)

        return image, target





def create_coco_dataset(data_root: str, image_set: str):
    """
    Create the coco dataset.
    :param data_root: The root directory of coco dataset.
    :param image_set: The image set, e.g. train, val.
    :return: The torch.utils.data.Dataset().
    """
    if image_set == "train":
        transforms = TrainTransform()
        root = os.path.join(data_root, "images/train2017")
        annFile = os.path.join(data_root, "annotations/instances_train2017.json")
    elif image_set == "val":
        transforms = ValTransform()
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
        center_x = int(bbox[0])
        center_y = int(bbox[1])
        width = int(bbox[2])
        height = int(bbox[3])
        print(label)
        cv2.rectangle(image, (center_x-width//2, center_y-height//2), (center_x+width//2, center_y+height//2),
                      (0, 0, 255))

    cv2.imshow("demo", image)
    cv2.waitKey()


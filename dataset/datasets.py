import os
import time

import torch
import torchvision
import torchvision.transforms.v2.functional as v2F
from PIL import Image
from pycocotools import mask
from torch.utils.data import Dataset
from torchvision import tv_tensors

from dataset.transforms import create_transform
from util.misc import list_of_dicts_to_dict_of_lists, show_image


def create_dataset(dataset_root: str, dataset_name: str):
    """
    Create training and validation datasets.
    :param dataset_root: root directory of all datasets
    :param dataset_name: select the dataset to use
    :return: dataset_train, dataset_val
    """
    if dataset_name == "coco":
        dataset_train = create_coco_dataset(os.path.join(dataset_root, dataset_name), "train")
        dataset_val = create_coco_dataset(os.path.join(dataset_root, dataset_name), "val")
    elif dataset_name == "visdrone":
        dataset_train = create_visdrone_dataset(os.path.join(dataset_root, dataset_name), "train")
        dataset_val = create_visdrone_dataset(os.path.join(dataset_root, dataset_name), "val")
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


def create_visdrone_dataset(data_root: str, image_set: str):
    """
    Create the visdrone dataset.
    :param data_root: The root directory of visdrone dataset.
    :param image_set: The image set, e.g. train, val, test.
    :return: The torch.utils.data.Dataset().
    """
    transform_train, transform_val = create_transform()

    root = os.path.join(data_root, image_set)
    if image_set == "train":
        transforms = transform_train
    elif image_set == "val":
        transforms = transform_val
    elif image_set == "test":
        transforms = transform_val
    else:
        raise ValueError("The image_set must be train, val or test.")

    return VisDroneDetection(root=root, transforms=transforms)


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
        canvas_size = tuple(v2F.get_size(image))

        batched_target = list_of_dicts_to_dict_of_lists(target['annotations'])
        target = {"image_id": target["image_id"], "origin_size": canvas_size}

        if "area" in batched_target:  # TODO: remove the line gt box
            for index in range(len(batched_target["area"])-1, -1, -1):
                if batched_target["area"][index] < 0.01:
                    for key in batched_target:
                        batched_target[key].pop(index)

        if "bbox" in batched_target and len(batched_target["bbox"]) > 0:
            bbox = batched_target["bbox"]
        else:
            bbox = torch.empty((0, 4))
        target["boxes"] = v2F.convert_bounding_box_format(
            tv_tensors.BoundingBoxes(
                bbox,
                format=tv_tensors.BoundingBoxFormat.XYWH,
                canvas_size=canvas_size,
                dtype=torch.float32,
            ),
            new_format=tv_tensors.BoundingBoxFormat.XYXY,
        )

        if "segmentation" in batched_target and len(batched_target["segmentation"]) > 0:
            target["masks"] = tv_tensors.Mask(
                torch.stack(
                    [
                        self.segmentation_to_mask(segmentation, canvas_size=canvas_size)
                        for segmentation in batched_target["segmentation"]
                    ]
                ),
            )
        else:
            target["masks"] = tv_tensors.Mask(
                torch.empty((0, canvas_size[0], canvas_size[1]), dtype=torch.uint8)
            )

        if "category_id" in batched_target and len(batched_target["category_id"]) > 0:
            target["labels"] = torch.tensor(batched_target["category_id"], dtype=torch.int64)
        else:
            target["labels"] = torch.empty(0, dtype=torch.int64)

        return image, target

    def segmentation_to_mask(self, segmentation, *, canvas_size):
        segmentation = (
            mask.frPyObjects(segmentation, *canvas_size)
            if isinstance(segmentation, dict)
            else mask.merge(mask.frPyObjects(segmentation, *canvas_size))
        )
        return torch.from_numpy(mask.decode(segmentation))


class VisDroneDetection(Dataset):
    """
    This class is used to load the VisDrone dataset for object detection.
    """
    def __init__(self, root, transforms=None):
        """
        :param root: The root directory of VisDrone dataset, which contains the subdirectories: images and annotations.
        :param transforms: A function/transform that takes input and target as entry and returns a transformed version.
        """
        super(VisDroneDetection, self).__init__()
        self.transforms = transforms
        self.img_root = os.path.join(root, "images")
        self.ann_root = os.path.join(root, "annotations")
        self.ids = sorted([os.path.splitext(f)[0] for f in os.listdir(self.img_root)])

        self.anns = dict()
        self.load_anns()

    def __getitem__(self, index):
        image = self.load_image(index)
        target = self.load_target(index)
        image, target = self.wrap_dataset_for_transforms_v2(image, target)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        return len(self.ids)

    def load_image(self, index):
        image_id = self.ids[index]
        image_path = os.path.join(self.img_root, image_id + ".jpg")
        return Image.open(image_path).convert("RGB")

    def load_target(self, index):
        image_id = self.ids[index]
        return self.anns[image_id]

    def load_anns(self):
        """
        load annotations from txt files
        """
        print("loading annotations into memory...")
        start_time = time.time()

        for image_id in self.ids:
            ann = {"image_id": image_id}
            boxes, scores, labels, truncations, occlusions = [], [], [], [], []

            file_path = os.path.join(self.ann_root, image_id + ".txt")
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    # split the line by ',' and remove the '\n' at the end of the line.
                    data = line.strip().split(',')[:8]
                    left, top, width, height, score, category, truncation, occlusion = [float(val) for val in data]

                    if width < 1 or height < 1 or category == 0 or category == 11:  # TODO: remove line gt box
                        continue

                    boxes.append([left, top, width, height])
                    labels.append(int(category))
                    scores.append(score)
                    truncations.append(int(truncation))
                    occlusions.append(int(occlusion))

            ann.update({'boxes': boxes, 'labels': labels, 'scores': scores, 'truncations': truncations,
                        'occlusions': occlusions})
            self.anns[image_id] = ann

        end_time = time.time()
        print("Done (t={:0.2f}s)".format(end_time - start_time))

    def wrap_dataset_for_transforms_v2(self, image, target):
        canvas_size = tuple(v2F.get_size(image))

        batched_target = target
        target = {"image_id": target["image_id"], "origin_size": canvas_size}

        if len(batched_target["boxes"]) == 0:
            bbox = torch.empty((0, 4))
        else:
            bbox = batched_target["boxes"]
        target["boxes"] = v2F.convert_bounding_box_format(
            tv_tensors.BoundingBoxes(
                bbox,
                format=tv_tensors.BoundingBoxFormat.XYWH,
                canvas_size=canvas_size,
                dtype=torch.float32,
            ),
            new_format=tv_tensors.BoundingBoxFormat.XYXY,
        )

        target["labels"] = torch.tensor(batched_target["labels"], dtype=torch.int64)
        target["scores"] = torch.tensor(batched_target["scores"], dtype=torch.float32)
        target["truncations"] = torch.tensor(batched_target["truncations"], dtype=torch.int32)
        target["occlusions"] = torch.tensor(batched_target["occlusions"], dtype=torch.int32)

        return image, target


if __name__ == "__main__":
    # demo for the create_dataset()
    dataset_train, dataset_val = create_dataset("./", "visdrone")
    for i in range(len(dataset_train)):
        image, target = dataset_train[i]
        show_image(image, target, "xyxy")

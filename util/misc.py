from collections import defaultdict

import cv2
import torch
from torchvision.transforms import v2


def collate_fn(batch):
    """
    Dynamic padding, the image inputs padding to the maximum size in the current batch.
    :param batch: Batch data. ( [(image, target), (image, target), ...] )
    :return:
    """
    batch = list(zip(*batch))  # [(image, image, ...), (target, target, ...)]
    images = list(batch[0])

    max_height = max([image.shape[1] for image in images])
    max_width = max([image.shape[2] for image in images])

    for i in range(len(images)):
        pad_height = max_height - images[i].shape[1]
        pad_width = max_width - images[i].shape[2]
        images[i] = v2.functional.pad(images[i], [0, 0, pad_width, pad_height])

    batch[0] = torch.stack(images, dim=0)
    return batch


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = defaultdict(list)
    for dct in list_of_dicts:
        for key, value in dct.items():
            dict_of_lists[key].append(value)
    return dict(dict_of_lists)


def show_image(image, target):
    """
    Show the image with bounding boxes.
    :param image: The image.
    :param target: The target in the image.
    """
    image = image.permute(1, 2, 0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for i in range(len(target['boxes'])):
        bbox = target['boxes'][i]
        label = target['labels'][i]
        center_x = int(bbox[0])
        center_y = int(bbox[1])
        width = int(bbox[2])
        height = int(bbox[3])
        cv2.putText(image, str(label.item()), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255))
        cv2.rectangle(image, (center_x - width // 2, center_y - height // 2),
                      (center_x + width // 2, center_y + height // 2),
                      (0, 0, 255))

    cv2.imshow("demo", image)
    cv2.waitKey()


def override_options(opts, checkpoint):
    """
    Override the options with the checkpoint.
    :param opts: The options.
    :param checkpoint: The checkpoint.
    :return:
    """
    # the keys need to be overridden
    keys = {'dataset_root', 'dataset_name', 'batch_size', 'epochs', 'lr', 'weight_decay', 'lr_drop', 'log_dir'}
    for key in keys:
        opts[key] = checkpoint['opts'][key]

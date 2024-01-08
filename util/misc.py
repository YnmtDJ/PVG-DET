import os
from collections import defaultdict

import cv2
import torch
from torchvision.ops import box_convert
from torchvision.transforms import v2


def collate_fn(batch):
    """
    Dynamic padding, the image inputs padding to the maximum size in the current batch.
    :param batch: Batch data. ( [(image, target), (image, target), ...] )
    :return:
    """
    batch = list(zip(*batch))  # [(image, image, ...), (target, target, ...)]
    return list(batch[0]), list(batch[1])
    # images = list(batch[0])
    #
    # max_height = max([image.shape[1] for image in images])
    # max_width = max([image.shape[2] for image in images])
    #
    # for i in range(len(images)):
    #     pad_height = max_height - images[i].shape[1]
    #     pad_width = max_width - images[i].shape[2]
    #     images[i] = v2.functional.pad(images[i], [0, 0, pad_width, pad_height])
    #
    # batch[0] = torch.stack(images, dim=0)
    # return batch


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = defaultdict(list)
    for dct in list_of_dicts:
        for key, value in dct.items():
            dict_of_lists[key].append(value)
    return dict(dict_of_lists)


def show_image(image, target, in_fmt):
    """
    Show the image with bounding boxes.
    :param image: The image.
    :param target: The target in the image.
    :param in_fmt: The format of the input bounding box, e.g. xyxy, xywh, cxcywh.
    """
    image = image.permute(1, 2, 0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for i in range(len(target['boxes'])):
        bbox = target['boxes'][i]
        label = target['labels'][i]
        bbox = box_convert(bbox, in_fmt, 'cxcywh')
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
    keys = {'dataset_root', 'dataset_name', 'batch_size', 'lr', 'weight_decay', 'lr_drop', 'log_dir'}
    for key in keys:
        setattr(opts, key, getattr(checkpoint['opts'], key))


def save_checkpoint(opts, model, optimizer, lr_scheduler, epoch):
    """
    Save the checkpoint
    :param opts: The options.
    :param model: The model.
    :param optimizer: The optimizer.
    :param lr_scheduler: The learning rate scheduler.
    :param epoch: The current epoch.
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'opts': opts
    }
    if not os.path.exists(os.path.dirname(opts.checkpoint_path)):  # check if the parent directory exists
        os.makedirs(os.path.dirname(opts.checkpoint_path))
    torch.save(checkpoint, opts.checkpoint_path)

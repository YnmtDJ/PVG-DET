import os
from collections import defaultdict
from functools import partial

import cv2
import torch
from torch.optim.lr_scheduler import LambdaLR
from torchvision import tv_tensors
from torchvision.ops import box_convert


def collate_fn(batch):
    """
    Dynamic padding, the image inputs padding to the maximum size in the current batch.
    :param batch: Batch data. ( [(image, target), (image, target), ...] )
    :return:
    """
    batch = list(zip(*batch))  # [(image, image, ...), (target, target, ...)]
    return list(batch[0]), list(batch[1])


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
    image = image.permute(1, 2, 0).cpu().numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for i in range(len(target['boxes'])):
        bbox = target['boxes'][i]
        label = target['labels'][i]
        bbox = box_convert(bbox, in_fmt, 'cxcywh')
        center_x, center_y, width, height = [int(val) for val in bbox]
        cv2.putText(image, str(label.item()), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
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
    keys = {'dataset_name', 'batch_size', 'epochs', 'warmup_epochs', 'lr'}
    for key in keys:
        setattr(opts, key, getattr(checkpoint['opts'], key))
    opts.start_epoch = checkpoint['epoch'] + 1


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


def fix_boxes(boxes: tv_tensors.BoundingBoxes):
    """
    Fix the boxes, make sure the boxes are in the correct format.
    :param boxes: The boxes to be fixed.
    """
    num, _ = boxes.shape
    eps = 1
    if boxes.format == tv_tensors.BoundingBoxFormat.XYXY:
        x_idx = torch.eq(boxes[:, 0], boxes[:, 2])
        y_idx = torch.eq(boxes[:, 1], boxes[:, 3])
    elif boxes.format == tv_tensors.BoundingBoxFormat.XYWH or boxes.format == tv_tensors.BoundingBoxFormat.CXCYWH:
        x_idx = torch.eq(torch.zeros([num]), boxes[:, 2])
        y_idx = torch.eq(torch.zeros([num]), boxes[:, 3])
    else:
        raise ValueError("Invalid bounding box format.")
    boxes[:, 2][x_idx] += eps
    boxes[:, 3][y_idx] += eps


def build_lr_scheduler(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Build the learning rate scheduler.
    :param optimizer: The optimizer.
    :param num_warmup_steps: The number of warmup steps.
    :param num_training_steps: The number of total training steps.
    :param last_epoch: The index of the last epoch when resuming training.
    :return: The learning rate scheduler.
    """
    def lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    warmup_lr_lambda = partial(
        lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, warmup_lr_lambda, last_epoch)

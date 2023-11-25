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


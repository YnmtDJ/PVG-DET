import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.datasets import create_coco_dataset, create_dataset
from model.criterion import SetCriterion
from model.de_gcn import DeGCN
from util.misc import collate_fn
from util.option import get_opts


def train_one_epoch(dataloader, model, criterion, optimizer, epoch, writer):
    """
    Train the model for one epoch.
    :param dataloader: The training dataloader.
    :param model: The detection model.
    :param criterion: The criterion for calculating the loss.
    :param optimizer: The optimizer for training model.
    :param epoch: Current epoch.
    :param writer: SummaryWriter for writing the log.
    """
    device = next(model.parameters()).device
    model.train()
    criterion.train()
    for i, (images, targets) in enumerate(dataloader):
        start_time = time.time()
        images = images.to(device)
        targets = [{k: v.to(device) if hasattr(v, 'to') else v for k, v in target.items()} for target in targets]
        outputs = model(images)
        loss, losses = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("train/loss", loss.item(), epoch * len(dataloader) + i)
        writer.add_scalar("train/loss_ce", losses['loss_ce'].item(), epoch * len(dataloader) + i)
        writer.add_scalar("train/loss_bbox", losses['loss_bbox'].item(), epoch * len(dataloader) + i)
        writer.add_scalar("train/loss_giou", losses['loss_giou'].item(), epoch * len(dataloader) + i)
        end_time = time.time()
        print("one iteration time: {:.2f}s".format(end_time - start_time))


if __name__ == "__main__":
    # get the options
    opts = get_opts()
    checkpoint = None
    if opts.checkpoint_path is not None and os.path.exists(opts.checkpoint_path):  # continue training
        checkpoint = torch.load(opts.checkpoint_path, map_location='cpu')
        opts = checkpoint['opts']

    # prepare the dataset
    dataset_train, dataset_val = create_dataset(opts.dataset_root, opts.dataset_name)
    if opts.dataset_name == "COCO":
        num_classes = 91  # because the coco dataset max label id is 90, so we set the num_classes to 91
    elif opts.dataset_name == "ImageNet":
        raise NotImplementedError("ImageNet dataset is not implemented yet.")
    else:
        num_classes = 20  # default num_classes
    dataloader_train = DataLoader(dataset_train, batch_size=opts.batch_size, shuffle=True, drop_last=False,
                                  collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=opts.batch_size, shuffle=False, drop_last=False,
                                collate_fn=collate_fn)

    # prepare the model, criterion, optimizer, lr_scheduler, writer
    device = torch.device(opts.device)
    model = DeGCN(num_classes).to(device)
    criterion = SetCriterion(num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opts.lr_drop)
    writer = SummaryWriter(opts.log_dir)

    # load the parameters to continue training
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        opts.start_epoch = checkpoint['epoch'] + 1

    print("Start training...")
    start_time = time.time()
    for epoch in range(opts.start_epoch, opts.epochs):
        # train for one epoch
        train_one_epoch(dataloader_train, model, criterion, optimizer, epoch, writer)
        lr_scheduler.step()

        # evaluate on the val dataset
        evaluate(dataloader_val, model, criterion, device)

        # save the checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'opts': opts
        }
        torch.save(checkpoint, opts.checkpoint_path)

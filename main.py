import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.datasets import create_dataset
from evaluate import evaluate_coco
from model import build
from train import train_one_epoch
from util.misc import collate_fn, override_options, save_checkpoint
from util.option import get_opts

if __name__ == "__main__":
    opts = get_opts()  # get the options
    checkpoint = None
    if opts.resume is not None:  # continue training
        checkpoint = torch.load(opts.resume, map_location='cpu')
        override_options(opts, checkpoint)  # override some options with the checkpoint

    # prepare the dataset
    dataset_train, dataset_val = create_dataset(opts.dataset_root, opts.dataset_name)
    dataloader_train = DataLoader(dataset_train, batch_size=opts.batch_size, shuffle=True, drop_last=False,
                                  collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=opts.batch_size, shuffle=False, drop_last=False,
                                collate_fn=collate_fn)

    # prepare the model, criterion, optimizer, lr_scheduler, writer
    model, criterion = build(opts)
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
    for epoch in range(opts.start_epoch, opts.epochs):
        print("epoch {} start...".format(epoch))
        start_time = time.time()
        try:
            # train for one epoch
            train_one_epoch(model, criterion, dataloader_train, optimizer, epoch, writer)
            lr_scheduler.step()  # update the learning rate

            # evaluate on the val dataset
            evaluate_coco(model, criterion, dataloader_val, epoch, writer)

        except Exception as e:
            raise e

        finally:
            # save the checkpoint
            save_checkpoint(opts, model, optimizer, lr_scheduler, epoch)
            end_time = time.time()
            print("epoch {} cost: {}s".format(epoch, end_time - start_time))

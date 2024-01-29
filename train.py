import torch
from tqdm import tqdm


def train_one_epoch(model, criterion, dataloader, optimizer, lr_scheduler, epoch, writer):
    """
    Train the model for one epoch.
    :param model: The detection model.
    :param criterion: The criterion for calculating the loss.
    :param dataloader: The training dataloader.
    :param optimizer: The optimizer for training model.
    :param lr_scheduler: The learning rate scheduler.
    :param epoch: Current epoch.
    :param writer: SummaryWriter for writing the log.
    """
    device = next(model.parameters()).device
    model.train()
    criterion.train()
    avg_loss, avg_loss_ce, avg_loss_bbox, avg_loss_ctrness = 0, 0, 0, 0
    for i, (images, targets) in enumerate(tqdm(dataloader)):
        if i % 200 == 0 and torch.cuda.is_available():  # TODO: really need it?
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) if hasattr(v, 'to') else v for k, v in target.items()} for target in targets]
        losses = model(images, targets)
        loss_ce = losses['classification']
        loss_bbox = losses['bbox_regression']
        loss_ctrness = losses['bbox_ctrness']
        loss = loss_ce + loss_bbox + loss_ctrness
        # loss, loss_ce, loss_bbox, loss_giou = criterion(outputs, targets)

        # back propagation and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()  # update the learning rate

        # update the average loss
        avg_loss = (avg_loss * i + loss.item()) / (i + 1)
        avg_loss_ce = (avg_loss_ce * i + loss_ce.item()) / (i + 1)
        avg_loss_bbox = (avg_loss_bbox * i + loss_bbox.item()) / (i + 1)
        avg_loss_ctrness = (avg_loss_ctrness * i + loss_ctrness.item()) / (i + 1)

        # TODO: delete it
        writer.add_scalar("test/loss", loss.item(), epoch*len(dataloader)+i)
        writer.add_scalar("test/loss_ce", loss_ce.item(), epoch*len(dataloader)+i)
        writer.add_scalar("test/loss_bbox", loss_bbox.item(), epoch*len(dataloader)+i)
        writer.add_scalar("test/loss_ctrness", loss_ctrness.item(), epoch*len(dataloader)+i)

    # write the loss to tensorboard
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/loss_ce", avg_loss_ce, epoch)
    writer.add_scalar("train/loss_bbox", avg_loss_bbox, epoch)
    writer.add_scalar("train/loss_ctrness", avg_loss_ctrness, epoch)
    # writer.add_scalar("train/loss_giou", loss_giou.item(), epoch)

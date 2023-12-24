

def train_one_epoch(model, criterion, dataloader, optimizer, epoch, writer):
    """
    Train the model for one epoch.
    :param model: The detection model.
    :param criterion: The criterion for calculating the loss.
    :param dataloader: The training dataloader.
    :param optimizer: The optimizer for training model.
    :param epoch: Current epoch.
    :param writer: SummaryWriter for writing the log.
    """
    device = next(model.parameters()).device
    model.train()
    criterion.train()
    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = [{k: v.to(device) if hasattr(v, 'to') else v for k, v in target.items()} for target in targets]
        outputs = model(images)
        loss, loss_ce, loss_bbox, loss_giou = criterion(outputs, targets)

        # back propagation and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # write the loss to tensorboard
        writer.add_scalar("train/loss", loss.item(), epoch * len(dataloader) + i)
        writer.add_scalar("train/loss_ce", loss_ce.item(), epoch * len(dataloader) + i)
        writer.add_scalar("train/loss_bbox", loss_bbox.item(), epoch * len(dataloader) + i)
        writer.add_scalar("train/loss_giou", loss_giou.item(), epoch * len(dataloader) + i)

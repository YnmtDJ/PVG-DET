import torch
from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, lr_scheduler, epoch, writer):
    """
    Train the model for one epoch.
    :param model: The detection model.
    :param dataloader: The training dataloader.
    :param optimizer: The optimizer for training model.
    :param lr_scheduler: The learning rate scheduler.
    :param epoch: Current epoch.
    :param writer: SummaryWriter for writing the log.
    """
    device = next(model.parameters()).device
    model.train()
    avg_loss = {}
    for i, (images, targets) in enumerate(tqdm(dataloader)):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) if hasattr(v, 'to') else v for k, v in target.items()} for target in targets]
        losses = model(images, targets)

        loss = torch.zeros(1, device=device)
        for k, v in losses.items():
            if k not in avg_loss:
                avg_loss[k] = 0.
            avg_loss[k] = (avg_loss[k] * i + v.item()) / (i + 1)  # update the average loss
            loss += v

        # back propagation and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()  # update the learning rate

    # write the loss to tensorboard
    for k, v in avg_loss.items():
        writer.add_scalar(f"train/loss_{k}", v, epoch)

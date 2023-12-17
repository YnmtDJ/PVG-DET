import torch
import torch.nn.functional as F
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_convert


@torch.no_grad()
def evaluate(dataset_name, model, criterion, dataloader, epoch, writer):
    """
    Evaluate the model on the dataset.
    :param dataset_name: The dataset name.
    :param model: The detection model.
    :param criterion: The criterion for calculating the loss.
    :param dataloader: The validation dataloader.
    :param epoch: Current epoch.
    :param writer: SummaryWriter for writing the log.
    """
    if dataset_name == "COCO":
        evaluate_coco(model, criterion, dataloader, epoch, writer)
    elif dataset_name == "ImageNet":
        raise NotImplementedError("ImageNet dataset is not implemented yet.")
    else:
        raise ValueError("Unknown dataset name.")


@torch.no_grad()
def evaluate_coco(model, criterion, dataloader, epoch, writer):
    """
    Evaluate the model on COCO dataset.
    :param model: The detection model.
    :param criterion: The criterion for calculating the loss.
    :param dataloader: The validation dataloader.
    :param epoch: Current epoch.
    :param writer: SummaryWriter for writing the log.
    """
    device = next(model.parameters()).device
    model.eval()
    criterion.eval()
    results = []
    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        outputs = model(images)
        loss, losses = criterion(outputs, targets)

        # write the loss to tensorboard
        writer.add_scalar("val/loss", loss.item(), epoch * len(dataloader) + i)
        writer.add_scalar("val/loss_ce", losses['loss_ce'].item(), epoch * len(dataloader) + i)
        writer.add_scalar("val/loss_bbox", losses['loss_bbox'].item(), epoch * len(dataloader) + i)
        writer.add_scalar("val/loss_giou", losses['loss_giou'].item(), epoch * len(dataloader) + i)

        # get the predict labels and scores
        predict_logits = outputs['pred_logits']  # (batch_size, num_queries, num_classes)
        batch_size, num_queries, num_classes = predict_logits.shape
        predict_prob = F.softmax(predict_logits, dim=-1)
        scores, labels = torch.max(predict_prob, dim=-1)

        # convert the boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2)
        predict_boxes = outputs['pred_boxes']  # (batch_size, num_queries, 4)
        boxes = box_convert(predict_boxes.reshape(-1, 4), "cxcywh", "xyxy")
        boxes = boxes.reshape(batch_size, num_queries, 4)

        for j, target in enumerate(targets):
            image_id = target['image_id']
            height, width = target["boxes"].canvas_size  # image size

            # non-normalized boxes coordinates by the size of each image
            boxes[j] = boxes[j] * torch.tensor([width, height, width, height], device=device)
            results.extend(
                [
                    {
                        "image_id": image_id,
                        "category_id": labels[j][k].item(),
                        "bbox": boxes[j][k].tolist(),
                        "score": scores[j][k].item(),
                    }
                    for k in range(num_queries)
                    if labels[j][k].item() != num_classes - 1  # filter the no-object class
                ]
            )

    # evaluate the results
    coco_gt = dataloader.dataset.coco
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    ap, ap50, ap75, aps, apm, apl, _, _, _, _, _, _ = coco_eval.stats[0]

    # write results to tensorboard
    writer.add_scalar("val/AP", ap, epoch)
    writer.add_scalar("val/AP50", ap50, epoch)
    writer.add_scalar("val/AP75", ap75, epoch)
    writer.add_scalar("val/APS", aps, epoch)
    writer.add_scalar("val/APM", apm, epoch)
    writer.add_scalar("val/APL", apl, epoch)

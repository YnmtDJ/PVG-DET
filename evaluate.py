import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_convert
from tqdm import tqdm

from util.visdrone_eval import eval_det


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
    elif dataset_name == "VisDrone":
        evaluate_visdrone(model, criterion, dataloader, epoch, writer)
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
    for i, (images, targets) in enumerate(tqdm(dataloader)):
        if i % 200 == 0 and torch.cuda.is_available():  # TODO: really need it?
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) if hasattr(v, 'to') else v for k, v in target.items()} for target in targets]
        predictions = model(images)

        for j, target in enumerate(targets):
            image_id = target['image_id']
            prediction = predictions[j]
            predict_num = prediction['labels'].shape[0]
            boxes = box_convert(prediction['boxes'], 'xyxy', 'xywh')
            results.extend(
                [
                    {
                        "image_id": image_id,
                        "category_id": prediction['labels'][k].item(),
                        "bbox": boxes[k].tolist(),
                        "score": prediction['scores'][k].item(),
                    }
                    for k in range(predict_num)
                ]
            )

        # loss, loss_ce, loss_bbox, loss_giou = criterion(outputs, targets)
        #
        # # write the loss to tensorboard
        # writer.add_scalar("val/loss", loss.item(), epoch * len(dataloader) + i)
        # writer.add_scalar("val/loss_ce", loss_ce.item(), epoch * len(dataloader) + i)
        # writer.add_scalar("val/loss_bbox", loss_bbox.item(), epoch * len(dataloader) + i)
        # writer.add_scalar("val/loss_giou", loss_giou.item(), epoch * len(dataloader) + i)
        #
        # # get the predict labels and scores
        # predict_logits = outputs['pred_logits']  # (batch_size, num_queries, num_classes)
        # batch_size, num_queries, num_classes = predict_logits.shape
        # predict_prob = F.softmax(predict_logits, dim=-1)
        # scores, labels = torch.max(predict_prob, dim=-1)
        #
        # # convert the boxes from (center_x, center_y, width, height) to (x1, y1, width, height)
        # predict_boxes = outputs['pred_boxes']  # (batch_size, num_queries, 4)
        # boxes = box_convert(predict_boxes.reshape(-1, 4), "cxcywh", "xywh")
        # boxes = boxes.reshape(batch_size, num_queries, 4)
        #
        # for j, target in enumerate(targets):
        #     image_id = target['image_id']
        #     height, width = target["origin_size"]  # image size
        #
        #     # non-normalized boxes coordinates by the size of each image
        #     boxes[j] = boxes[j] * torch.tensor([width, height, width, height], device=device)
        #     results.extend(
        #         [
        #             {
        #                 "image_id": image_id,
        #                 "category_id": labels[j][k].item(),
        #                 "bbox": boxes[j][k].tolist(),
        #                 "score": scores[j][k].item(),
        #             }
        #             for k in range(num_queries)
        #             if labels[j][k].item() != num_classes - 1  # filter the no-object class
        #         ]
        #     )

    if len(results) == 0:  # model does not detect any object
        print("epoch: {}, evaluate coco, model does not detect any object.".format(epoch))
        return

    # evaluate the results
    coco_gt = dataloader.dataset.coco
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    ap, ap50, ap75, aps, apm, apl, _, _, _, _, _, _ = coco_eval.stats

    # write results to tensorboard
    writer.add_scalar("val/AP", ap, epoch)
    writer.add_scalar("val/AP50", ap50, epoch)
    writer.add_scalar("val/AP75", ap75, epoch)
    writer.add_scalar("val/APS", aps, epoch)
    writer.add_scalar("val/APM", apm, epoch)
    writer.add_scalar("val/APL", apl, epoch)


@torch.no_grad()
def evaluate_visdrone(model, criterion, dataloader, epoch, writer):
    """
    Evaluate the model on VisDrone dataset.
    :param model: The detection model.
    :param criterion: The criterion for calculating the loss.
    :param dataloader: The validation dataloader.
    :param epoch: Current epoch.
    :param writer: SummaryWriter for writing the log.
    """
    device = next(model.parameters()).device
    model.eval()
    criterion.eval()
    all_gt, all_det, all_height, all_width = [], [], [], []
    for i, (images, targets) in enumerate(tqdm(dataloader)):
        if i % 200 == 0 and torch.cuda.is_available():  # TODO: really need it?
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) if hasattr(v, 'to') else v for k, v in target.items()} for target in targets]
        predictions = model(images)

        for j, target in enumerate(targets):
            height, width = target["origin_size"]  # image size
            prediction = predictions[j]
            predict_num = prediction['labels'].shape[0]

            # convert the boxes format
            target_boxes = box_convert(target['boxes'], 'xyxy', 'xywh')
            predict_boxes = box_convert(prediction['boxes'], 'xyxy', 'xywh')

            gt = torch.cat([target_boxes, target["scores"].unsqueeze(-1), target["labels"].unsqueeze(-1),
                            target["truncations"].unsqueeze(-1), target["occlusions"].unsqueeze(-1)],
                           dim=1).cpu().numpy().astype(np.int32)

            const = -torch.ones([predict_num, 1], dtype=torch.int32).to(device)  # constant tensor -1
            det = torch.cat([predict_boxes, prediction["scores"].unsqueeze(-1), prediction["labels"].unsqueeze(-1),
                             const, const], dim=1).cpu().numpy()

            all_gt.append(gt)
            all_det.append(det)
            all_height.append(height)
            all_width.append(width)

    # evaluate the results
    ap_all, ap_50, ap_75, ar_1, ar_10, ar_100, ar_500 = eval_det(all_gt, all_det, all_height, all_width)

    # write results to tensorboard
    writer.add_scalar("val/AP", ap_all, epoch)
    writer.add_scalar("val/AP50", ap_50, epoch)
    writer.add_scalar("val/AP75", ap_75, epoch)
    writer.add_scalar("val/AR1", ar_1, epoch)
    writer.add_scalar("val/AR10", ar_10, epoch)
    writer.add_scalar("val/AR100", ar_100, epoch)
    writer.add_scalar("val/AR500", ar_500, epoch)

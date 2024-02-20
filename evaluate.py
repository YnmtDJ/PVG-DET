import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_convert
from tqdm import tqdm

from util.visdrone_eval import eval_det


@torch.no_grad()
def evaluate(dataset_name, model, dataloader, epoch, writer):
    """
    Evaluate the model on the dataset.
    :param dataset_name: The dataset name.
    :param model: The detection model.
    :param dataloader: The validation dataloader.
    :param epoch: Current epoch.
    :param writer: SummaryWriter for writing the log.
    """
    if dataset_name == "COCO":
        evaluate_coco(model, dataloader, epoch, writer)
    elif dataset_name == "VisDrone":
        evaluate_visdrone(model, dataloader, epoch, writer)
    elif dataset_name == "ImageNet":
        raise NotImplementedError("ImageNet dataset is not implemented yet.")
    else:
        raise ValueError("Unknown dataset name.")


@torch.no_grad()
def evaluate_coco(model, dataloader, epoch, writer):
    """
    Evaluate the model on COCO dataset.
    :param model: The detection model.
    :param dataloader: The validation dataloader.
    :param epoch: Current epoch.
    :param writer: SummaryWriter for writing the log.
    """
    device = next(model.parameters()).device
    model.eval()
    results = []
    for i, (images, targets) in enumerate(tqdm(dataloader)):
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
def evaluate_visdrone(model, dataloader, epoch, writer):
    """
    Evaluate the model on VisDrone dataset.
    :param model: The detection model.
    :param dataloader: The validation dataloader.
    :param epoch: Current epoch.
    :param writer: SummaryWriter for writing the log.
    """
    device = next(model.parameters()).device
    model.eval()
    all_gt, all_det, all_height, all_width = [], [], [], []
    for i, (images, targets) in enumerate(tqdm(dataloader)):
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

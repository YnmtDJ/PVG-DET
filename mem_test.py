import random
import time
from collections import Counter

import numpy as np
import torch
from matplotlib import pyplot as plt
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import box_convert
from tqdm import tqdm

from dataset.datasets import create_dataset, create_visdrone_dataset
from evaluate import evaluate_coco
from model import build_retinanet, build_fcos
from model.fcos.fcos import FCOS, FCOSHead, FCOSClassificationHead, FCOSRegressionHead, FCOS_ResNet50_FPN_Weights
from util.misc import collate_fn, build_lr_scheduler, show_image
from util.option import get_opts
from util.visdrone_eval import eval_det


def func():
    opts = get_opts()  # get the options
    opts.device = "cuda"
    device = torch.device(opts.device)
    opts.dataset_name = "VisDrone"
    model = torch.load("c:\\users\\16243\\Downloads\\model.pth")
    # model = build_retinanet(opts)
    # optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    # lr_scheduler = build_lr_scheduler(optimizer, warmup_epochs=opts.warmup_epochs, epochs=opts.epochs)

    # model.load_state_dict(checkpoint['model'])
    model.train()
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    # opts.start_epoch = checkpoint['epoch'] + 1

    # demo for the create_dataset()
    dataset_train, dataset_val = create_dataset("./dataset", "VisDrone")
    dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, drop_last=False, collate_fn=collate_fn)

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader_train):

            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) if hasattr(v, 'to') else v for k, v in target.items()} for target in targets]

            losses = model(images, targets)
            # loss_ce = losses['classification']
            # loss_bbox = losses['bbox_regression']
            # print("loss_ce: ", loss_ce.item())
            # print("loss_bbox: ", loss_bbox.item())
            # print("---------------------------")
            # for j in range(4):
                # image = images[j]
                # prediction = predictions[j]
                # show_image(image, prediction, "xyxy")


def fun1():
    opts = get_opts()  # get the options
    opts.device = "cpu"
    device = torch.device(opts.device)
    opts.dataset_name = "VisDrone"
    checkpoint = torch.load("checkpoint/visdrone/2024_01_24.pth", map_location='cpu')
    model = build_retinanet(opts)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    # demo for the create_dataset()
    dataset_train, dataset_val = create_dataset("./dataset", "VisDrone")
    dataloader_val = DataLoader(dataset_val, batch_size=4, shuffle=False, drop_last=False, collate_fn=collate_fn)

    all_gt, all_det, all_height, all_width = [], [], [], []
    for i, (images, targets) in enumerate(dataloader_val):

        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) if hasattr(v, 'to') else v for k, v in target.items()} for target in targets]
        # predictions = model(images)

        for j, target in enumerate(targets):
            height, width = target["origin_size"]  # image size
            # prediction = predictions[j]
            # predict_num = prediction['labels'].shape[0]

            # convert the boxes format
            target_boxes = box_convert(target['boxes'], 'xyxy', 'xywh')
            # predict_boxes = box_convert(prediction['boxes'], 'xyxy', 'xywh')

            gt = torch.cat([target_boxes, target["scores"].unsqueeze(-1), target["labels"].unsqueeze(-1),
                            target["truncations"].unsqueeze(-1), target["occlusions"].unsqueeze(-1)],
                           dim=1).cpu().numpy().astype(np.int32)

            all_gt.append(gt)
            all_det.append(gt)
            all_height.append(height)
            all_width.append(width)

    # evaluate the results
    ap_all, ap_50, ap_75, ar_1, ar_10, ar_100, ar_500 = eval_det(all_gt, all_det, all_height, all_width)
    print("...")


def func2():
    dataset_train, dataset_val = create_dataset("./dataset", "VisDrone")
    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, drop_last=False, collate_fn=collate_fn)
    # transform = GeneralizedRCNNTransform([256, 272, 288, 304, 320, 336, 352], 512, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    areas = []
    for i, (images, targets) in enumerate(tqdm(dataloader_train)):
        # images, targets = transform(images, targets)
        target = targets[0]
        height, width = target['origin_size']
        boxes = box_convert(target["boxes"], 'xyxy', 'xywh')
        areas.extend([round(val, 4) for val in torch.sqrt((boxes[:, 2]*boxes[:, 3])/(width*height)).tolist()])

    counts = Counter(areas)
    # 将Counter结果转换为列表形式
    counts = list(counts.items())
    # 将键转换为字符串并进行排序
    sorted_counts = sorted(counts, key=lambda x: str(x[0]))

    # 创建条形图
    plt.bar([str(count[0]) for count in sorted_counts], [count[1] for count in sorted_counts])
    plt.xlabel('areas')
    plt.ylabel('Count')
    plt.title('Value Distribution')
    plt.show()


def func3():
    opts = get_opts()  # get the options
    opts.device = "cpu"
    device = torch.device(opts.device)
    opts.dataset_name = "VisDrone"

    model = build_fcos(opts)
    model.load_state_dict(torch.load("C:\\Users\\hu.nan\\Downloads\\visdrone\\6e-4+warmup\\2+4\\checkpoint.pth", map_location="cpu")["model"])
    model.eval()

    dataset_train, dataset_val = create_dataset("./dataset", "VisDrone")
    dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, drop_last=False, collate_fn=collate_fn)

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader_train):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) if hasattr(v, 'to') else v for k, v in target.items()} for target in targets]
            predictions = model(images, targets)
            for j in range(4):
                image = images[j]
                prediction = predictions[j]
                show_image(image, prediction, "xyxy")


def test_for_evaluate():
    dataset_test = create_visdrone_dataset("dataset/VisDrone/", "test")
    dataloader_test = DataLoader(dataset_test, batch_size=4, shuffle=False, drop_last=False, collate_fn=collate_fn)

    opts = get_opts()  # get the options
    opts.device = "cuda"
    device = torch.device(opts.device)
    opts.dataset_name = "VisDrone"

    model = build_fcos(opts)
    model.load_state_dict(torch.load("C:\\Users\\16243\\Downloads\\checkpoint.pth")["model"])
    model.eval()

    with torch.no_grad():
        all_gt, all_det, all_height, all_width = [], [], [], []
        for i, (images, targets) in enumerate(tqdm(dataloader_test)):
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
        print("evaluate finish.")


if __name__ == '__main__':
    test_for_evaluate()

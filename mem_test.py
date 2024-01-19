import math
import time
from argparse import Namespace
from collections import Counter

import torch
from matplotlib import pyplot as plt
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.datasets import create_dataset
from dataset.transforms import create_transform
from evaluate import evaluate
from model.vig.gcn.edge import pairwise_distance
from model.vig.vig import ViG
from util.misc import collate_fn, override_options, save_checkpoint, show_image
from model import build, build_retinanet
from util.option import get_opts


def func():
    opts = get_opts()  # get the options
    opts.device = "cpu"
    opts.dataset_name = "VisDrone"
    model = build_retinanet(opts)
    model.load_state_dict(torch.load("c:\\users\\hu.nan\\Downloads\\vig_retinanet (1).pth", map_location="cpu")['model'])
    model.eval()
    # demo for the create_dataset()
    dataset_train, dataset_val = create_dataset("./dataset", "VisDrone")
    dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, drop_last=False, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=4, shuffle=False, drop_last=False, collate_fn=collate_fn)

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader_train):
            predictions = model(images)
            for j in range(4):
                image = images[j]
                prediction = predictions[j]
                show_image(image, prediction, "xyxy")


if __name__ == '__main__':
    func()

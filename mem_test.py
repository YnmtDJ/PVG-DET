import torch
from torch.utils.data import DataLoader

from dataset.datasets import create_dataset
from model import build_retinanet
from util.misc import collate_fn
from util.option import get_opts


def func():
    opts = get_opts()  # get the options
    opts.device = "cuda"
    device = torch.device(opts.device)
    opts.dataset_name = "VisDrone"
    checkpoint = torch.load("c:\\users\\16243\\Downloads\\vig_retinanet.pth")
    model = build_retinanet(opts)
    model.load_state_dict(checkpoint['model'])
    model.train()
    # demo for the create_dataset()
    dataset_train, dataset_val = create_dataset("./dataset", "VisDrone")
    dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, drop_last=False, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=4, shuffle=False, drop_last=False, collate_fn=collate_fn)

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader_train):

            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) if hasattr(v, 'to') else v for k, v in target.items()} for target in targets]

            losses = model(images, targets)
            loss_ce = losses['classification']
            loss_bbox = losses['bbox_regression']
            print("loss_ce: ", loss_ce.item())
            print("loss_bbox: ", loss_bbox.item())
            print("---------------------------")
            # for j in range(4):
                # image = images[j]
                # prediction = predictions[j]
                # show_image(image, prediction, "xyxy")


if __name__ == '__main__':
    func()

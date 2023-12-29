import torch
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image
import torchvision.transforms.v2 as v2
from dataset.transforms import create_transform
from model.vig.vig import ViG


def func():
    transformer_train = v2.Resize(200)
    images = Image(torch.zeros(1, 3, 100, 100))
    boxes = BoundingBoxes(
        [[49, 49, 50, 50]],
        format=BoundingBoxFormat.CXCYWH,
        canvas_size=(100, 100)
    )
    images, boxes = transformer_train(images, boxes)
    print(images, boxes)


if __name__ == '__main__':
    func()

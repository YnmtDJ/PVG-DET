import torchvision
import torch

from dataset.dataset import create_dataset
from util.option import get_opts

if __name__ == "__main__":

    opts = get_opts()

    dataset = create_dataset(opts)


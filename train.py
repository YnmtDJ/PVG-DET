import os

from torch.utils.data import DataLoader

from dataset.dataset import create_coco_dataset
from model.criterion import SetCriterion
from model.de_gcn import DeGCN
from util.misc import collate_fn
from util.option import get_opts

if __name__ == "__main__":
    opts = get_opts()  # get the options

    # prepare the dataset
    if opts.datasetName == "COCO":
        dataset_train = create_coco_dataset(os.path.join(opts.datasetRoot, opts.datasetName), "train")
        dataset_val = create_coco_dataset(os.path.join(opts.datasetRoot, opts.datasetName), "val")
        num_classes = 91  # because the coco dataset max label id is 90, so we set the num_classes to 91
    elif opts.datasetName == "ImageNet":
        raise NotImplementedError("ImageNet dataset is not implemented yet.")
    else:
        dataset_train = None
        dataset_val = None
        num_classes = 20  # default num_classes

    dataloader_train = DataLoader(dataset_train, batch_size=opts.batchSize, shuffle=True, drop_last=False,
                                  collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=opts.batchSize, shuffle=False, drop_last=False,
                                collate_fn=collate_fn)

    model = DeGCN(num_classes)
    criterion = SetCriterion(num_classes)

    for i, (images, targets) in enumerate(dataloader_val):
        predict = model(images)
        print(predict.shape)

import os

from torch.utils.data import DataLoader

from dataset.dataset import create_coco_dataset
from model.vig.vig import ViG
from util.misc import collate_fn
from util.option import get_opts

if __name__ == "__main__":

    opts = get_opts()

    coco_root = os.path.join(opts.datasetRoot, opts.datasetName, "images/val2017")
    coco_annFile = os.path.join(opts.datasetRoot, opts.datasetName, "annotations/instances_val2017.json")
    dataset = create_coco_dataset(coco_root, coco_annFile, True)
    dataloader = DataLoader(dataset, batch_size=opts.batchSize, shuffle=True, drop_last=False, collate_fn=collate_fn)

    model = ViG()

    for i, (images, targets) in enumerate(dataloader):
        predict = model(images)
        print(predict.shape)
    
    



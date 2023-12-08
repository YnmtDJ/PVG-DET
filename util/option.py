import argparse


def get_opts():
    parser = argparse.ArgumentParser(description="GCN_DETECTION")

    # dataset parameters
    parser.add_argument("--datasetRoot", type=str, default="dataset", help="the root directory of dataset")
    parser.add_argument("--datasetName", type=str, default="COCO", help="select the dataset to use")

    # model parameters

    # training parameters
    parser.add_argument("--batchSize", type=int, default=8, help="batch size")
    parser.add_argument('--epochs', type=int, default=100, help="number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--checkpointPath', type=str, default="checkpoint", help="the path to save the checkpoint")

    return parser.parse_args()

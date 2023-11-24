import argparse


def get_opts():
    parser = argparse.ArgumentParser(description="GCN_DETECTION")

    # dataset parameters
    parser.add_argument("--datasetRoot", type=str, default="dataset", help="the root directory of dataset")
    parser.add_argument("--datasetName", type=str, default="COCO", help="select the dataset to use")

    # model parameters

    # training parameters
    parser.add_argument("--batchSize", type=int, default=8, help="batch size")

    return parser.parse_args()

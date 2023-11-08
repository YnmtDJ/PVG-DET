import argparse


def get_opts():
    parser = argparse.ArgumentParser(description="GCN_DETECTION")

    # dataset parameters
    parser.add_argument("--datasetRoot", type=str, default="dataset", help="the root directory of dataset")
    parser.add_argument("--datasetName", type=str, default="COCO", help="select the dataset to use")

    return parser.parse_args()

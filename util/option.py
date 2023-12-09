import argparse


def get_opts():
    parser = argparse.ArgumentParser(description="GCN_DETECTION")

    # dataset parameters
    parser.add_argument("--dataset_root", type=str, default="dataset", help="the root directory of dataset")
    parser.add_argument("--dataset_name", type=str, default="COCO", help="select the dataset to use")

    # model parameters

    # training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument('--epochs', type=int, default=100, help="number of training epochs")
    parser.add_argument("--start_epoch", type=int, default=0, help='start epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="optimizer weight decay")
    parser.add_argument('--lr_drop', type=int, default=200, help="learning rate decay step")
    parser.add_argument('--device', type=str, default='cuda', help='device to use for training / testing')
    parser.add_argument('--checkpoint_path', type=str, help="the path to save the checkpoint. "
                                                            "if not None, continue training from the checkpoint and "
                                                            "ignore some command-line parameters.")
    parser.add_argument('--log_dir', type=str, help="the directory to save the log file")

    return parser.parse_args()

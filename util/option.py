import argparse


def get_opts():
    parser = argparse.ArgumentParser(description="GCN_DETECTION")

    # dataset parameters
    parser.add_argument("--dataset_root", type=str, default="dataset", help="root directory of all datasets")
    parser.add_argument("--dataset_name", type=str, default="COCO", help="select the dataset to use")

    # model parameters

    # training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument('--epochs', type=int, default=12, help="number of training epochs")
    parser.add_argument("--start_epoch", type=int, default=0, help='start epoch')
    parser.add_argument("--warm_up_epochs", type=int, default=2, help="number of warm-up epochs")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--device', type=str, default='cuda', help='device to use for training / testing')
    parser.add_argument('--checkpoint_path', type=str, required=True, help="the path to save the checkpoint.")
    parser.add_argument('--resume', type=str, help="the path to the checkpoint to resume training from, "
                                                   "this will override some command-line parameters")
    parser.add_argument('--log_dir', type=str, required=True, help="the directory to save the log file")

    return parser.parse_args()

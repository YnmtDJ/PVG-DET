import argparse


def get_opts():
    parser = argparse.ArgumentParser(description="GCN_DETECTION")

    # dataset parameters
    parser.add_argument("--dataset_root", type=str, default="dataset", help="root directory of all datasets")
    parser.add_argument("--dataset_name", type=str, default="VisDrone", help="select the dataset to use")

    # model parameters
    parser.add_argument("--backbone", type=str, default="pvg_s", help="the backbone of the model")
    parser.add_argument("--num_classes", type=int, required=True, help="the number of classes in the dataset")
    parser.add_argument("--min_size", type=int, default=800, help="minimum size of the image to be rescaled "
                                                                  "before feeding it to the model")
    parser.add_argument("--max_size", type=int, default=1333, help="maximum size of the image to be rescaled "
                                                                   "before feeding it to the model")
    parser.add_argument("--k", type=int, default=9, help="the number of neighbors in ViG")
    parser.add_argument("--gcn", type=str, default="MRConv2d", help="the graph convolution type in ViG "
                                                                    "(MRConv2d, EdgeConv2d, GraphSAGE, GINConv2d)")
    parser.add_argument("--drop_prob", type=float, default=0.1, help="the probability of DropPath in ViG")

    # training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument('--epochs', type=int, default=18, help="number of training epochs")
    parser.add_argument("--start_epoch", type=int, default=0, help='start epoch')
    parser.add_argument("--warmup_epochs", type=int, default=2, help="number of warm-up epochs")
    parser.add_argument('--lr', type=float, default=6e-4, help="learning rate")
    parser.add_argument('--device', type=str, default='cuda', help='device to use for training / testing')
    parser.add_argument('--checkpoint_path', type=str, required=True, help="the path to save the checkpoint.")
    parser.add_argument('--resume', type=str, help="the path to the checkpoint to resume training from, "
                                                   "this will override some command-line parameters")
    parser.add_argument('--log_dir', type=str, required=True, help="the directory to save the log file")

    return parser.parse_args()

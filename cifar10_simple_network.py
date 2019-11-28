import argparse

# Parser configuration
# -------------------------

parser = argparse.ArgumentParser(description="Script to train a simple CIFAR10 network")
parser.add_argument(
    "-e", "--epochs", help="Number of epochs. Default is 20", type=int, default=20
)
parser.add_argument(
    "-b", "--batch-size", help="Batch size. Default is 128", type=int, default=128
)
parser.add_argument(
    "-d",
    "--dropout",
    help="Percentage of dropout. Default is 0.4",
    type=float,
    default=0.4,
)
parser.add_argument(
    "-v",
    "--verbose",
    help="If set, output details of the execution",
    action="store_true",
)
parser.add_argument(
    "-w",
    "--weights",
    help="h5 file from which load (if the file exists) and save the model weights. Default is cifar10_simple_model.h5",
    type=str,
    default="cifar10_simple_model.h5",
)
parser.add_argument(
    "-p",
    "--path",
    help="Begin path of training results. Files <path>_accuracy.png and <path>_loss.png will be created. Default is train_results",
    type=str,
    default="train_results",
)
parser.add_argument(
    "-g",
    "--gpu",
    help="The ID of the GPU to use. If not set, no GPU configuration is done. Default is None",
    type=int,
    default=None,
)

# Global parameters
# -------------------------

nb_classes = 10
input_shape = (32, 32, 3)

args = parser.parse_args()
epochs = args.epochs
batch_size = args.batch_size
dropout = args.dropout
verbose = args.verbose
path_weights = args.weights
path_results = args.path
gpu_id = args.gpu

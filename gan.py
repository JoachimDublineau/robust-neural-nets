import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.models import Model, load_model

import src

tf.keras.backend.clear_session()

# Parser configuration
# -------------------------

parser = argparse.ArgumentParser(
    description="Script to train a Generative Adversarial Network (GAN) for CIFAR10"
)
parser.add_argument(
    "-e", "--epochs", help="Number of epochs. Default is 20", type=int, default=20
)
parser.add_argument(
    "-b", "--batch-size", help="Batch size. Default is 128", type=int, default=128
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
    help="h5 file from which load (if it exists) and save the model weights. Default is {}gan_model.h5".format(
        src.models_dir
    ),
    type=str,
    default="{}gan_model.h5".format(src.models_dir),
)
parser.add_argument(
    "--output-log",
    help="Output file name of training info. Default is {}gan_training_log.csv".format(
        src.results_dir
    ),
    type=str,
    default="{}gan_training_log.csv".format(src.results_dir),
)
parser.add_argument(
    "--gpu",
    help="The ID of the GPU (ordered by PCI_BUS_ID) to use. If not set, no GPU configuration is done. Default is None",
    type=int,
    default=None,
)
parser.add_argument(
    "--tf-log-level",
    help="Tensorflow minimum cpp log level. Default is 0",
    choices=["0", "1", "2", "3"],
    default="0",
)

# Global parameters
# -------------------------

args = parser.parse_args()
epochs = args.epochs
batch_size = args.batch_size
verbose = args.verbose
path_weights = args.weights
output_log = args.output_log
gpu_id = args.gpu

os.environ["TF_CPP_MIN_LOG_LEVEL"] = args.tf_log_level

src.create_dir_if_not_found(src.models_dir)
src.create_dir_if_not_found(src.results_dir)

# GPU configuration
# -------------------------

if gpu_id is not None:
    if verbose:
        print("GPU configuration...")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if verbose:
        print("GPU configuration done.")


# Get data
# -------------------------


if verbose:
    print("Getting data...")

x_train, y_train, x_test, y_test = src.cifar10.load_data()

x_train = (x_train.astype("float32") - 127.5) / 127.5
x_test = (x_test.astype("float32") - 127.5) / 127.5

y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(src.cifar10.labels))
y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(src.cifar10.labels))

if verbose:
    print("Data is loaded.")

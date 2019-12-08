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

# Build model
# -------------------------


def build_generator_model():
    """ Generator model. Transforms random vector of size (100,) into a picture
    """
    generator_input = layers.Input(shape=(100,))

    model = layers.Dense(8 * 8 * 128, use_bias=False)(generator_input)
    model = layers.BatchNormalization()(model)
    model = layers.LeakyReLU()(model)

    model = layers.Reshape((8, 8, 128))(model)

    # Upsmalpling: shape (None, 8, 8, 128) to shape (None, 8, 8, 64)
    model = layers.Conv2DTranspose(
        64, (5, 5), strides=(1, 1), padding="same", use_bias=False
    )(model)
    model = layers.BatchNormalization()(model)
    model = layers.LeakyReLU()(model)

    # Upsmalpling: shape (None, 8, 8, 64) to shape (None, 16, 16, 32)
    model = layers.Conv2DTranspose(
        32, (5, 5), strides=(2, 2), padding="same", use_bias=False
    )(model)
    model = layers.BatchNormalization()(model)
    model = layers.LeakyReLU()(model)

    # Upsmalpling: shape (None, 16, 16, 32) to shape (None, 32, 32, 3)
    model = layers.Conv2DTranspose(
        3, (5, 5), strides=(2, 2), padding="same", use_bias=False
    )(model)

    generator_model = Model(generator_input, model)
    return generator_model


def build_discriminator_model():
    """ Discriminator model. Tells if a picture is real or artificial
    """
    discriminator_input = layers.Input(shape=(32, 32, 3))

    model = layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same")(
        discriminator_input
    )
    model = layers.LeakyReLU()(model)
    model = layers.Dropout(0.3)(model)

    model = layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")(model)
    model = layers.LeakyReLU()(model)
    model = layers.Dropout(0.3)(model)

    model = layers.Flatten()(model)
    model = layers.Dense(1)(model)

    discriminator_model = Model(discriminator_input, model)
    return discriminator_model


# Train model
# -------------------------

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as klayers
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model, load_model

tf.keras.backend.clear_session()

# Parser configuration
# -------------------------

parser = argparse.ArgumentParser(
    description="Script to train a simple CIFAR10 network")
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

# Simple network
# -------------------------


def build_simple_network():
    # TODO

    # Get data
    # -------------------------


if verbose:
    print("Getting data...")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

y_train = tf.keras.utils.to_categorical(y_train, num_classes=nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=nb_classes)

# Build model
# -------------------------

if verbose:
    print("Building model...")

model = build_simple_network()

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

if verbose:
    print(model.summary())

# Train model
# -------------------------

if verbose:
    print("Training model...")

if path_weights is not None and os.path.exists(path_weights):
    model.load_weights(path_weights)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
    epochs=epochs,
    verbose=int(verbose),
)

if verbose:
    print("Saving weights...")

model.save(path_weights)

# Saving graphs
# -------------------------

if verbose:
    print("Saving graphs...")

plt.plot(history.history["accuracy"], color="c")
plt.plot(history.history["val_accuracy"], color="r")
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Valid"], loc="upper right")
plt.savefig("{}_accuracy.png".format(path_results), dpi=400)
plt.clf()

plt.plot(history.history["loss"], color="c")
plt.plot(history.history["val_loss"], color="r")
plt.title("Model loss (categorical crossentropy)")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Valid"], loc="upper right")
plt.savefig("{}_loss.png".format(path_results), dpi=400)

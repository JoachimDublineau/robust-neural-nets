import argparse
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.models import load_model
import tqdm
from tensorflow.keras.utils import Sequence
import tensorflow.keras.callbacks as cbks

import src

tf.keras.backend.clear_session()

# argparse specific type
# -------------------------

def zero_one_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError('{} not a float'.format(x))

    if not 0 <= x <= 1:
        raise argparse.ArgumentTypeError('{} not in range [0, 1]'.format(x))

    return x

# Train methods functions
# -------------------------


def train_method_simple():
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        epochs=epochs,
        verbose=int(verbose),
        callbacks=[csv_logger, checkpoint],
    )

    if verbose:
        print("Model is trained.")


def train_method_defense_fgsm():
    # get optimizer and loss function
    optimizer = model.optimizer
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # create metrics
    train_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    batch_accuracy = tf.keras.metrics.CategoricalAccuracy()
    validation_loss = tf.keras.metrics.Mean()
    validation_accuracy = tf.keras.metrics.CategoricalAccuracy()

    # store metrics related to an epoch to make the process easier
    epoch_metrics = {
        'loss': train_loss,
        'accuracy': train_accuracy,
        'val_loss': validation_loss,
        'val_accuracy': validation_accuracy,
    }

    # create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_train.shape[0]).batch(batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    # prepare callbacks
    callback_metrics = list(epoch_metrics.keys())
    callback_model = model._get_callback_model()
    callback_model.stop_training = False
    logger_params = {
        'epochs': epochs,
        'steps': None,
        'verbose': int(verbose),
        'batch_size': batch_size,
        'samples': x_train.shape[0],
        'do_validation': True,
        'metrics': callback_metrics,
    }

    base_logger = cbks.BaseLogger()
    callbacks = [base_logger, csv_logger, checkpoint]

    if verbose:
        progressbar_callback = cbks.ProgbarLogger(count_mode='samples')
        callbacks.insert(1, progressbar_callback)

    for callback in callbacks:
        callback.set_params(logger_params)
        callback.validation_data = validation_dataset
        callback.set_model(callback_model)
        callback.on_train_begin()

    # epoch loop
    for epoch in range(epochs):
        epoch_logs = {}

        for callback in callbacks:
            callback.on_epoch_begin(epoch)

        # batch loop
        for batch_index, (batch_x, batch_y) in enumerate(train_dataset):
            batch_logs = {'batch': batch_index, 'size': batch_x.shape[0]}

            for callback in callbacks:
                callback.on_batch_begin(batch_index, logs=batch_logs)

            # compute signed gradients for batch x and create adversarial images
            batch_x_signed_gradients = src.attacks.compute_signed_gradients(batch_x, batch_y, model, tf.keras.losses.categorical_crossentropy, batch_size=batch_size, verbose=False)
            batch_x_adversarial = tf.keras.backend.clip(batch_x + epsilon * batch_x_signed_gradients, 0, 1)

            # forward step
            with tf.GradientTape() as tape:
                predictions = model(batch_x)
                predictions_adversarials = model(batch_x_adversarial)

                loss_value = loss_fn(batch_y, predictions)
                loss_value_adversarial = loss_fn(batch_y, predictions_adversarials)

                loss_value = alpha * loss_value + (1 - alpha) * loss_value_adversarial

            # backward step
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # update metrics
            train_loss(loss_value)
            train_accuracy(batch_y, predictions)
            batch_accuracy(batch_y, predictions)

            # update batch logs
            batch_logs['loss'] = float(loss_value)
            batch_logs['accuracy'] = float(batch_accuracy.result())

            # reset batch metric
            batch_accuracy.reset_states()

            for callback in callbacks:
                callback.on_batch_end(batch_index, logs=batch_logs)

        # validation batch loop
        for batch_index, (batch_x, batch_y) in enumerate(validation_dataset):
            predictions = model(batch_x)
            loss_value = loss_fn(batch_y, predictions)

            # update metrics
            validation_loss(loss_value)
            validation_accuracy(batch_y, predictions)

        # update epochs logs and reset epoch metrics
        for metric_name, metric in epoch_metrics.items():
            epoch_logs[metric_name] = float(metric.result())
            metric.reset_states()

        for callback in callbacks:
            callback.on_epoch_end(epoch, logs=epoch_logs)

    for callback in callbacks:
        callback.on_train_end()

train_methods = {
    "simple": train_method_simple,
    "defense-fgsm": train_method_defense_fgsm
}

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
    help="h5 file from which load (if it exists) and save the model weights. Default is {}cifar10_simple_model.h5".format(
        src.models_dir
    ),
    type=str,
    default="{}cifar10_simple_model.h5".format(src.models_dir),
)
parser.add_argument(
    "--output-log",
    help="Output file name of training info. Default is {}training_log.csv".format(
        src.results_dir
    ),
    type=str,
    default="{}training_log.csv".format(src.results_dir),
)
parser.add_argument(
    "--gpu",
    help="The ID of the GPU (ordered by PCI_BUS_ID) to use. If not set, no GPU configuration is done. Default is None",
    type=int,
    default=None,
)
parser.add_argument("--train-method", default="simple",
                    choices=train_methods.keys(), help="The train method to use. Default is simple")
parser.add_argument("--epsilon", type=float, help="The value of epsilon to use while training with 'defense_fgsm' traning method")
parser.add_argument("--alpha", type=zero_one_float, help="The value of alpha to use while training with 'defense_fgsm' traning method. Must be in range [0, 1]. Default is 0.5", default=0.5)
parser.add_argument('--tf-log-level', default='3',
                    choices=['0', '1', '2', '3'], help='Tensorflow minimum cpp log level. Default is 3')

# Global parameters
# -------------------------

args = parser.parse_args()
epochs = args.epochs
batch_size = args.batch_size
dropout = args.dropout
verbose = args.verbose
path_weights = args.weights
output_log = args.output_log
gpu_id = args.gpu
train_method = args.train_method
epsilon = args.epsilon
alpha = args.alpha

os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_log_level

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

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

y_train = tf.keras.utils.to_categorical(
    y_train, num_classes=len(src.cifar10.labels))
y_test = tf.keras.utils.to_categorical(
    y_test, num_classes=len(src.cifar10.labels))

if verbose:
    print("Data is loaded.")

# Build model
# -------------------------

if verbose:
    print("Building model...")

model = src.cifar10.build_simple_network(dropout)

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

if verbose:
    print(model.summary())
    print("Model is built.")

# Train model
# -------------------------

if verbose:
    print("Training model...")

if path_weights is not None and os.path.exists(path_weights):
    model.load_weights(path_weights)

csv_logger = CSVLogger(output_log, append=True, separator=";")
checkpoint = ModelCheckpoint(
    path_weights, verbose=int(verbose), save_freq="epoch")

train_methods[train_method]()

if verbose:
    print("Saving weights...")

model.save(path_weights)

if verbose:
    print("Weights are saved.")

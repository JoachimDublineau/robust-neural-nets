import tensorflow.keras.layers as klayers
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model, load_model

channels = 3
height = 32
width = 32
input_shape = (height, width, channels)

labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_data():
    """ Function to load CIFAR10 data. If data is not in ~/.keras/datasets/ it will be downloaded

    Returns:
        x_train (np_array): training tensors
        y_train (np_array): training labels
        x_test (np_array): testing tensors
        y_test (np_array): testing labels
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return x_train, y_train, x_test, y_test


# Simple network
# -------------------------


def build_simple_network(dropout):
    """ Function to build a simple network for CIFAR10

    Args:
        dropout (int): the value of dropout

    Returns:
        cifar10_network: the final model
    """
    X = klayers.Input(input_shape)

    network = klayers.Conv2D(
        32,
        activation=None,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-3),
    )(X)
    network = klayers.BatchNormalization()(network)
    network = klayers.Activation("relu")(network)
    network = klayers.Dropout(dropout)(network)

    network = klayers.Conv2D(
        32,
        activation=None,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-3),
    )(network)
    network = klayers.BatchNormalization()(network)
    network = klayers.Activation("relu")(network)
    network = klayers.Dropout(dropout)(network)

    network = klayers.Conv2D(
        32,
        activation=None,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-3),
    )(network)
    network = klayers.BatchNormalization()(network)
    network = klayers.Activation("relu")(network)
    network = klayers.Dropout(dropout)(network)

    network = klayers.AveragePooling2D()(network)
    network = klayers.Flatten()(network)
    network = klayers.Dense(len(labels), activation="softmax")(network)

    cifar10_network = Model(inputs=X, outputs=network)
    return cifar10_network

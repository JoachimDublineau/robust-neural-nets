import tensorflow.keras.layers as klayers
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.models import Model, load_model

nb_classes = 10
channels = 3
height = 32
width = 32
input_shape = (height, width, channels)

# Simple network
# -------------------------


def build_simple_network(dropout):
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
    network = klayers.Dense(nb_classes, activation="softmax")(network)
    return Model(inputs=X, outputs=network)

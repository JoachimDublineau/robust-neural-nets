import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model, load_model

import src

tf.keras.backend.clear_session()

# Parser configuration
# -------------------------

parser = argparse.ArgumentParser(
    description="Script to train a Generative Adversarial Network (GAN) for CIFAR10 on 1 class"
)
parser.add_argument(
    "-e", "--epochs", help="Number of epochs. Default is 20", type=int, default=20
)
parser.add_argument(
    "-b", "--batch-size", help="Batch size. Default is 128", type=int, default=128
)
parser.add_argument(
    "--random-vec",
    help="The size of the random vector given to generator model. Default is 100",
    type=int,
    default=100,
)
parser.add_argument(
    "-c",
    "--choosen-class",
    help="The class on which the GAN will be trained. Default is 8",
    type=int,
    default=8,
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
    "--save-img",
    help="If set generated images will be saved after each epoch under {}gan_generated_images_<epoch>.png".format(
        src.results_dir
    ),
    action="store_true",
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
random_vec_size = args.random_vec
choosen_class = args.choosen_class
verbose = args.verbose
path_weights = args.weights
save_img = args.save_img
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

x_train, y_train, _, _ = src.cifar10.load_data()
# We choose pictures of 1 one class
x_train = x_train[y_train.flatten() == choosen_class]
x_train = (x_train.astype("float32") - 127.5) / 127.5  # Scale to [-1, 1]

if verbose:
    print("Data is loaded.")

# Build model
# -------------------------

if verbose:
    print("Building model...")


def build_generator_model():
    """ Generator model. Transforms random vector into a picture
    """
    generator_input = layers.Input(shape=(random_vec_size,))

    model = layers.Dense(8 * 8 * 128, use_bias=False)(generator_input)
    model = layers.BatchNormalization()(model)
    model = layers.LeakyReLU()(model)

    model = layers.Reshape((8, 8, 128))(model)

    # Upsampling: shape (None, 8, 8, 128) to shape (None, 8, 8, 64)
    model = layers.Conv2DTranspose(
        64, (5, 5), strides=(1, 1), padding="same", use_bias=False
    )(model)
    model = layers.BatchNormalization()(model)
    model = layers.LeakyReLU()(model)

    # Upsampling: shape (None, 8, 8, 64) to shape (None, 16, 16, 32)
    model = layers.Conv2DTranspose(
        32, (5, 5), strides=(2, 2), padding="same", use_bias=False
    )(model)
    model = layers.BatchNormalization()(model)
    model = layers.LeakyReLU()(model)

    # Upsampling: shape (None, 16, 16, 32) to shape (None, 32, 32, 3)
    model = layers.Conv2DTranspose(
        3, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"
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
    model = layers.Dense(1, activation="sigmoid")(model)

    discriminator_model = Model(discriminator_input, model)
    return discriminator_model


# Building discriminator model
discriminator = build_discriminator_model()
discriminator.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    metrics=["accuracy"],
)
discriminator.trainable = False

# Building generator model
generator = build_generator_model()

# Building GAN model
random_vec = layers.Input(shape=(random_vec_size,))
decision = discriminator(generator(random_vec))

gan = Model(inputs=random_vec, outputs=decision)
gan.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=1e-4))

if verbose:
    print("Model is built.")
    print(gan.summary())

# Training
# -------------------------

if verbose:
    print("Training GAN model...")

if path_weights is not None and os.path.exists(path_weights):
    gan.load_weights(path_weights)

# Adversarial ground truths
real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

iterations = len(x_train) // batch_size

for e in range(epochs):
    if verbose:
        print("Epoch {}/{} :".format(e, epochs), end="")

    for i in range(iterations):

        # Training discriminator
        # -------------------------

        discriminator.trainable = True

        # We get real and generated images
        real_images = x_train[i * batch_size : (i + 1) * batch_size]
        random_vecs = np.random.normal(
            loc=0, scale=1, size=(batch_size, random_vec_size)
        )
        fake_images = generator.predict_on_batch(random_vecs)

        # We train and get the total loss
        discr_real_metrics = discriminator.train_on_batch(x=real_images, y=real)
        discr_fake_metrics = discriminator.train_on_batch(x=fake_images, y=fake)
        discriminator_loss = 0.5 * (discr_real_metrics[0] + discr_fake_metrics[0])
        discriminator_accuracy = 0.5 * (discr_real_metrics[1] + discr_fake_metrics[1])

        if verbose and i == iterations - 1:
            print(
                " discriminator_loss: {} - discriminator_accuracy: {} - ".format(
                    discriminator_loss, discriminator_accuracy
                ),
                end="",
            )

        # Training generator
        # -------------------------

        discriminator.trainable = False
        generator_loss = gan.train_on_batch(x=random_vecs, y=real)
        if verbose and i == iterations - 1:
            print("generator_loss: {}".format(generator_loss))

    gan.save(path_weights)

    # Saving 5 fake pictures
    # -------------------------

    if save_img:
        fig, axs = plt.subplots(1, 5)

        for f in range(5):
            random_vec = np.random.normal(loc=0, scale=1, size=(1, random_vec_size))
            fake_image = generator.predict_on_batch(random_vec)[0]
            fake_image = 0.5 * fake_image + 0.5  # Rescale

            axs[f].imshow(fake_image)
            axs[f].axis("off")

        fig.savefig("{}gan_generated_images_{}.png".format(src.results_dir, e))
        plt.close()

if verbose:
    print("Model is trained.")

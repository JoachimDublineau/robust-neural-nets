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
    description="Script to train a Conditional Generative Adversarial Network (CGAN) for CIFAR10"
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
    help="If set, 1 generated image of each class will be saved after each epoch under {}gan_generated_images_<epoch>.png".format(
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

x_train, y_train, x_test, y_test = src.cifar10.load_data()

x_train = (x_train.astype("float32") - 127.5) / 127.5  # Scale to [-1, 1]
x_test = (x_test.astype("float32") - 127.5) / 127.5  # Scale to [-1, 1]

y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(src.cifar10.labels))
y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(src.cifar10.labels))

if verbose:
    print("Data is loaded.")

# Build model
# -------------------------

if verbose:
    print("Building model...")


def build_generator_model():
    """ Generator model. Transforms random vector into a picture
    """
    random_vec_input = layers.Input(shape=(random_vec_size,))
    labels_input = layers.Input(shape=(len(src.cifar10.labels),))
    generator_input = layers.Concatenate()([random_vec_input, labels_input])

    model = layers.Dense(8 * 8 * 512)(generator_input)
    model = layers.BatchNormalization()(model)
    model = layers.LeakyReLU()(model)

    model = layers.Reshape((8, 8, 512))(model)

    # Upsampling: shape (None, 8, 8, 512) to shape (None, 8, 8, 256)
    model = layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding="same")(model)
    model = layers.BatchNormalization()(model)
    model = layers.LeakyReLU()(model)

    # Upsampling: shape (None, 8, 8, 256) to shape (None, 8, 8, 128)
    model = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same")(model)
    model = layers.BatchNormalization()(model)
    model = layers.LeakyReLU()(model)

    # Upsampling: shape (None, 8, 8, 128) to shape (None, 16, 16, 64)
    model = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same")(model)
    model = layers.BatchNormalization()(model)
    model = layers.LeakyReLU()(model)

    # Upsampling: shape (None, 16, 16, 64) to shape (None, 32, 32, 3)
    model = layers.Conv2DTranspose(
        3, (5, 5), strides=(2, 2), padding="same", activation="tanh"
    )(model)

    generator_model = Model(inputs=[random_vec_input, labels_input], outputs=model)
    return generator_model


def build_discriminator_model():
    """ Discriminator model. Tells if a picture is real or artificial
    """
    image_input = layers.Input(shape=(32, 32, 3))
    labels_input = layers.Input(shape=(len(src.cifar10.labels),))

    model = layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same")(image_input)
    model = layers.BatchNormalization()(model)
    model = layers.LeakyReLU()(model)
    model = layers.Dropout(0.3)(model)

    model = layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")(model)
    model = layers.BatchNormalization()(model)
    model = layers.LeakyReLU()(model)
    model = layers.Dropout(0.3)(model)

    model = layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same")(model)
    model = layers.BatchNormalization()(model)
    model = layers.LeakyReLU()(model)
    model = layers.Dropout(0.3)(model)

    model = layers.Conv2D(512, (5, 5), strides=(2, 2), padding="same")(model)
    model = layers.BatchNormalization()(model)
    model = layers.LeakyReLU()(model)
    model = layers.Dropout(0.3)(model)

    model = layers.Flatten()(model)
    model = layers.Concatenate()([model, labels_input])
    model = layers.Dense(1, activation="sigmoid")(model)

    discriminator_model = Model(inputs=[image_input, labels_input], outputs=model)
    return discriminator_model


# Inputs: random vector and labels
random_vec_input = layers.Input(shape=(random_vec_size,))
labels_input = layers.Input(shape=(len(src.cifar10.labels),))

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
decision = discriminator([generator([random_vec_input, labels_input]), labels_input])

gan = Model(inputs=[random_vec_input, labels_input], outputs=decision)
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

        # We get real images from the dataset
        real_images = x_train[i * batch_size : (i + 1) * batch_size]
        real_labels = y_train[i * batch_size : (i + 1) * batch_size]

        # We get generated images from the generator
        random_vecs = np.random.normal(
            loc=0, scale=1, size=(batch_size, random_vec_size)
        )
        fake_labels = tf.keras.utils.to_categorical(
            np.random.randint(0, len(src.cifar10.labels), batch_size)
        )
        fake_images = generator.predict_on_batch([random_vecs, fake_labels])

        # We train and get the total loss
        discr_real_metrics = discriminator.train_on_batch(
            x=[real_images, real_labels], y=real
        )
        discr_fake_metrics = discriminator.train_on_batch(
            x=[fake_images, fake_labels], y=fake
        )
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
        generator_loss = gan.train_on_batch(x=[random_vecs, fake_labels], y=real)
        if verbose and i == iterations - 1:
            print("generator_loss: {}".format(generator_loss))

    gan.save(path_weights)

    # Saving 5 fake pictures
    # -------------------------

    if save_img:
        fig, axs = plt.subplots(2, 5)
        img_label = 0

        for k in range(2):
            for l in range(5):
                random_vec = np.random.normal(loc=0, scale=1, size=(1, random_vec_size))
                label = tf.keras.utils.to_categorical(
                    np.array([img_label]), num_classes=len(src.cifar10.labels)
                )
                fake_image = generator.predict_on_batch([random_vec, label])[0]
                fake_image = 0.5 * fake_image + 0.5  # Rescale

                axs[k, l].title.set_text(src.cifar10.labels[img_label])
                axs[k, l].imshow(fake_image)
                axs[k, l].axis("off")
                img_label += 1

        fig.savefig("{}gan_generated_images_{}.png".format(src.results_dir, e))
        plt.close()

if verbose:
    print("Model is trained.")

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tqdm


def compute_signed_gradients(x, y, model, loss, batch_size=128, verbose=False):
    """
    Compute the signed gradients of the loss with respect to x

    Args:
        x: the inputs
        y: the labels
        model: the model use to predict the inputs labels
        loss: the loss for which compute the gradients
        batch_size: the batch size used to compute signed gradients
        verbose: if True, output details on the execution
    """
    # Cast inputs into tensor
    def kast(element): return K.cast(element, x.dtype)
    x = kast(x)
    def kast(element): return K.cast(element, y.dtype)
    y = kast(y)

    signed_gradients = np.empty((0, 32, 32, 3))
    nb_batch = len(x) // batch_size + 1

    iterator = range(nb_batch)
    if verbose:
        iterator = tqdm.tqdm(
            iterator, desc='Computing signed gradients on batch', unit='batch')

    for batch_index in iterator:
        batch_x = x[batch_index * batch_size: (batch_index + 1) * batch_size]
        batch_y = y[batch_index * batch_size: (batch_index + 1) * batch_size]

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(batch_x)
            y_predicted = model(batch_x)
            loss_result = loss(batch_y, y_predicted)

        gradients = tape.gradient(loss_result, batch_x)
        signed_gradients = np.concatenate(
            (signed_gradients, tf.sign(gradients)))

    return signed_gradients

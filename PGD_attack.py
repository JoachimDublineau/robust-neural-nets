import numpy as np
from keras import backend as K 
from keras.losses import categorical_crossentropy
import tensorflow as tf
import src

def generate_perturbation(img_size, eps, norm = 2):
    """
    Generates a random perturbation of the given size. The norm
    (l2 or l_inf) of this pertubation is inferior to eps.
    INPUTS:
    - img_size: iterable giving the dimension of the perturbation.
    - eps: float defining the upper bound of the norms.
    - norm: 'l2' or 'inf' depending on what norm do you want.
    OUTPUTS:
    - perturbation: numpy array of dimension img_size.
    COMPUTATION TIME:
    In n*p
    """
    perturbation = np.random.random(size = img_size)
    pert_norm = np.linalg.norm(perturbation, ord=norm)
    if pert_norm > eps:
        perturbation *= np.random.random(eps)/pert_norm
    return perturbation

# # Test generate_perturbation
# perturbation = generate_perturbation((10,10), 1, norm = "l2")
# print(perturbation)
# print(np.linalg.norm(perturbation))

def compute_grad(model, loss, image, label):
    """
    Inspired from tensorflow core. 
    INPUTS:
    - model: tensorflow model
    - loss: tensorflow loss
    - image: reference image
    - label: one-hot encoding of the class
    OUTPUTS:
    - signed_grad: tensorflow tensor representing the gradient.
    Its dimension is equal to the one of image.
    COMPUTATION TIME:
    Fast
    """
    kast = lambda element: K.cast(element, dtype='float32')
    x = kast([image])
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x)
        loss_ = loss(label, prediction)
    gradient = tape.gradient(loss_, x)
    signed_grad = tf.sign(gradient)
    return signed_grad[0]

def projection(point, ref, eps):
    """
    Computes the projection of the point vector on the ball
    centered in ref of radius eps.
    INPUTS:
    - point: array representing the point the we want to project
    - ref: center of the ball on which the projection will be 
    done
    - eps: radius of the ball
    OUTPUT:
    - projection: array of the same shape of point and ref 
    COMPUTATION TIME:
    Very Fast
    """
    dist = np.linalg.norm(ref - point)
    if dist < eps:
        return point
    return eps*(point - ref)/dist 

# # Test projection:
# a = np.array([2,2], dtype = np.float32)
# b = np.array([1,1], dtype = np.float32)
# print(a-b)
# print(np.linalg.norm(a-b, ord=2))
# print(projection(a, b, 1))

def generate_pgd_attack(model, loss, ref_image, y_image, eps, 
                        norm = 2, nb_it = 10):
    """
    Computes a pgd attack for a given ref_image, model and loss.
    INPUTS:
    - model: tensorflow.keras model compiled with the loss and having
    input shape = ref_image.shape.
    - loss: element of keras.losses or customized loss respecting
    the same structure.
    - ref_image: array representing the input image.
    - y_image: label of the input image.
    - eps: maximal norm of the attack.
    - norm: argument for the calculation of the norm with 
    numpy.linalg.norm can be 2 or 'inf'. (Unused for now)
    - nb_it: number of iterations for the calculation of the PGD 
    attack.
    OUTPUTS:
    - curr_perturbation: array of the same shape as ref_image 
    representing the pdg attack.
    COMPUTATION TIME:
    Fast.
    """
    curr_perturbation = eps*compute_grad(model, loss, 
                                         ref_image, y_image)
    curr_perturbation = projection(curr_perturbation, 
                                   ref_image, eps)
    for iter in range(nb_it):
        curr_perturbated_img = ref_image + curr_perturbation
        signed_grad = compute_grad(model, loss, curr_perturbated_img, 
                                   y_image)
        curr_perturbation += eps*signed_grad
        curr_perturbation = projection(curr_perturbation, ref_image, 
                                       eps)
    return curr_perturbation

# Test PGD Attack:
x_train, y_train, x_test, y_test = src.cifar10.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

y_train = tf.keras.utils.to_categorical(y_train, \
    num_classes = len(src.cifar10.labels))
y_test = tf.keras.utils.to_categorical(y_test, \
    num_classes = len(src.cifar10.labels))

image = x_train[0]
label = y_train[0]
model = tf.keras.models.load_model("models/cifar10_simple_model_73_acc.h5")
perturbation = generate_pgd_attack(model, categorical_crossentropy, 
                             image, label, 1)
print("Perturbation:")
# print(perturbation)
print("Norm:", np.linalg.norm(perturbation))
print()
print("Image:")
# print(image)
print("Norm:", np.linalg.norm(image))
print("Model prediction:", np.argmax(model(K.cast([image], 
                                           dtype = 'float32'))[0]))
print()
print("Perturbated image:")
print("Norm:", np.linalg.norm(image + perturbation))
print("Model prediction:", np.argmax(model(K.cast([image + perturbation], 
                                        dtype = 'float32'))[0]))
print()

import matplotlib.pyplot as plt
fig=plt.figure(figsize=(1, 3))
fig.add_subplot(1, 3, 1)
plt.imshow(image)
fig.add_subplot(1, 3, 2)
plt.imshow(perturbation)
fig.add_subplot(1, 3, 3)
plt.imshow(image + perturbation)
plt.show()

## CODING THE BATCH Calculation : 
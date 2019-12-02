import numpy as np
from keras import backend as K 
from keras.losses import categorical_crossentropy
import tensorflow as tf

def generate_perturbation(img_size, eps, norm = "l2"):
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
    pert_norm = 0
    if norm == "l2":
        pert_norm = np.linalg.norm(perturbation)
    else:
        pert_norm = np.linalg.norm(perturbation, 'inf')
    if pert_norm > eps:
        perturbation *= np.random.random(eps)/pert_norm
    return perturbation

# # Test generate_perturbation
# perturbation = generate_perturbation((10,10), 1, norm = "l2")
# print(perturbation)
# print(np.linalg.norm(perturbation))

def compute_grad(model, loss, image, input_label):
    """
    Inspired from tensorflow core. 
    INPUTS:
    - model: tensorflow model
    - loss: tensorflow loss
    - image: reference image
    - input_label: one-hot encoding of the class
    - norm: 'l2' or 'inf' depending on what norm do you want.
    OUTPUTS:
    - perturbation: numpy array of dimension img_size.
    COMPUTATION TIME:
    In n*p
    """
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss_ = loss(input_label, prediction)
    gradient = tape.gradient(loss_, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

def projection(point, ref, eps, norm):
    dist = np.linalg.norm(ref - point, ord = norm)
    if dist < eps:
        return point
    return ref + eps*(point - ref)/dist 

# # Test projection:
# a = np.array([1,1], dtype = np.float32)
# b = np.array([0,0], dtype = np.float32)
# print(a-b)
# print(np.linalg.norm(a-b, ord=2))
# print(projection(a, b, 1, 2))

def maximize_loss(model, loss, ref_image, y_image, rate, eps, 
                  norm = 2, nb_it = 10):
    curr_perturbation = rate*compute_grad(model, loss, ref_image, y_image)
    for iter in range(nb_it):
        curr_perturbated_img = ref_image + curr_perturbation
        signed_grad = compute_grad(model, loss, curr_perturbated_img, y_image)
        curr_perturbation += projection(rate*signed_grad,ref_image,eps,norm)
    return curr_perturbation


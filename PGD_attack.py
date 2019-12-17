import numpy as np
from tensorflow.keras import backend as K 
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow as tf
import src
import time

def generate_perturbation(img_size, eps):
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
    pert_norm = np.linalg.norm(perturbation)
    if pert_norm > eps:
        perturbation *= np.random.random(eps)/pert_norm
    return perturbation

def compute_grad(model, loss, images, labels):
    """
    Inspired from tensorflow core. 
    INPUTS:
    - model: tensorflow model
    - loss: tensorflow loss
    - images: reference images
    - labels: one-hot encoding of the class
    OUTPUTS:
    - signed_grad: tensorflow tensor representing the gradient.
    Its dimension is equal to the one of image.
    COMPUTATION TIME:
    Fast
    """
    kast = lambda element: K.cast(element, dtype='float32')
    x = kast(images)
    with tf.GradientTape() as tape:
        tape.watch(x)
        predictions = model(x)
        loss_ = loss(labels, predictions)
    gradients = tape.gradient(loss_, x)
    signed_grads = tf.sign(gradients)
    return signed_grads

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

def generate_pgd_attack(model, loss, ref_image, y_image, eps, 
                        step = 0.1, threshold = 1e-3, nb_it_max = 20):
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
    - step: float gradient step
    - threshold: float convergence threshold
    - nb_it_max: max number of iterations.
    OUTPUTS:
    - curr_perturbation: array of the same shape as ref_image 
    representing the pdg attack.
    COMPUTATION TIME:
    Fast.
    """
    curr_perturbation = generate_perturbation(ref_image.shape, eps)
    signed_grad = compute_grad(model, loss, [ref_image], y_image)[0]
    dist = threshold+1
    i = 0 
    while dist > threshold: 
        i+=1     
        prev_perturbation = curr_perturbation
        curr_perturbation += step*signed_grad
        curr_perturbation = projection(curr_perturbation, ref_image, eps)
        dist = np.linalg.norm(prev_perturbation - curr_perturbation)
        if i > nb_it_max: break
    return curr_perturbation

def projection_on_batch(points, ref, eps):
    """
    Computes the projection of the point vector on the ball
    centered in ref of radius eps.
    INPUTS:
    - points: array representing the points the we want to project
    - ref: center of the ball on which the projection will be 
    done
    - eps: radius of the ball
    OUTPUT:
    - projections: array of the same shape of points 
    COMPUTATION TIME:
    Very Fast
    """
    projections = []
    for i, point in enumerate(points):
        dist = np.linalg.norm(ref[i] - point)
        if dist < eps:
            projections.append(point)
        else:
            projections.append(eps*(point - ref[i])/dist)
    return np.array(projections) 

def generate_pgd_attack_on_batch(model, loss, ref_images, y_images, eps, 
                                 batch_size, step = 0.1, threshold=1e-3, 
                                 nb_it_max = 20):
    """
    Computes a pgd attack for given ref_images, model and loss.
    INPUTS:
    - model: tensorflow.keras model compiled with the loss and having
    input shape = ref_image.shape.
    - loss: element of keras.losses or customized loss respecting
    the same structure.
    - ref_images: array representing the input images.
    - y_images: labels of the input images.
    - eps: maximal norm of the attack.
    - batch_size: int, size of ref_images
    - step: float gradient step
    - threshold: float convergence threshold 
    - nb_it_max: int max number of iterations
    OUTPUTS:
    - curr_perturbation: array of the same shape as ref_image 
    representing the pdg attack.
    COMPUTATION TIME:
    Fast.
    """
    curr_perturbations = [generate_perturbation(ref_images[0].shape,eps) for i in range(batch_size)]
    curr_perturbations = np.array(curr_perturbations)
    signed_grads = compute_grad(model, loss, ref_images, y_images)
    dist = threshold + 1
    count = 0
    while dist > threshold:
        count += 1
        prev_perturbations = curr_perturbations        
        curr_perturbations += step*signed_grads
        curr_perturbations = projection_on_batch(curr_perturbations, ref_images, eps)
        diff = curr_perturbations - prev_perturbations
        dist = np.linalg.norm(diff)/batch_size
        if count > nb_it_max: break
    return curr_perturbations


def generate_pgd_attack_on_batch_accelerated(model, loss, ref_images, y_images, eps, 
                                 batch_size, step = 0.1, nb_it = 5):
    """
    Computes a pgd attack for given ref_images, model and loss.
    INPUTS:
    - model: tensorflow.keras model compiled with the loss and having
    input shape = ref_image.shape.
    - loss: element of keras.losses or customized loss respecting
    the same structure.
    - ref_images: array representing the input images.
    - y_images: labels of the input images.
    - eps: maximal norm of the attack.
    - batch_size: int, size of ref_images
    - step: float gradient step
    - nb_it: int number of iterations
    OUTPUTS:
    - curr_perturbation: array of the same shape as ref_image 
    representing the pdg attack.
    COMPUTATION TIME:
    Fast.
    """
    curr_perturbations = [generate_perturbation(ref_images[0].shape,eps) for i in range(batch_size)]
    curr_perturbations = np.array(curr_perturbations)
    signed_grads = compute_grad(model, loss, ref_images, y_images)
    for i in range(nb_it):      
        curr_perturbations += step*signed_grads
        curr_perturbations = projection_on_batch(curr_perturbations, ref_images, eps)
    return curr_perturbations

def generate_pgd_attacks(model, loss, x, y, eps, batch_size,
                         step = 0.1, threshold=1e-3, nb_it_max = 20,
                         accelerated = False):
    """
    Computes a pgd attacks for a given x, model and loss.
    INPUTS:
    - model: tensorflow.keras model compiled with the loss and having
    input shape = ref_image.shape.
    - loss: element of keras.losses or customized loss respecting
    the same structure.
    - x: array representing the input images.
    - y: label of the input images.
    - eps: maximal norm of the attack.
    - step: float gradient step
    - threshold: float convergence threshold 
    - nb_it_max: float max number of iterations
    attack.
    OUTPUTS:
    - tab_perturbations: array of the same shape as x 
    representing the pdg attacks.
    COMPUTATION TIME:
    Fast proportionnal to the number of images.
    """
    tab_perturbations = []
    nb_batch = len(x) // batch_size + 1
    for i in range(nb_batch):
        end = 0
        if i == nb_batch - 1:
            end = len(x)
        else:
            end = (i+1)*batch_size
        x_batch = x[i*batch_size:end]
        y_batch = y[i*batch_size:end]
        size = end - i*batch_size
        if size == 0:
            break
        print("Computing Attacks for batch", i, " Images", i*batch_size, "to", end)
        if accelerated:
            perturbations = generate_pgd_attack_on_batch_accelerated(model, loss, x_batch,
                y_batch, eps, size, step, nb_it = 3)
        if not accelerated:
            perturbations = generate_pgd_attack_on_batch(model, loss, x_batch,
                y_batch, eps, size, step, threshold, nb_it_max)
        for perturbation in perturbations:
            tab_perturbations.append(perturbation)
    tab_perturbations = np.array(tab_perturbations, dtype = np.float32)
    return tab_perturbations

import numpy as np
from keras import backend as K 
from keras.losses import categorical_crossentropy

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

# Test generate_perturbation
perturbation = generate_perturbation((10,10), 1, norm = "l2")
print(perturbation)
print(np.linalg.norm(perturbation))

def maximize_loss(model, loss, ref_image, rate, eps, norm = "l2", 
                  nb_it = 10):
    size = ref_image.shape
    prev_perturbation = ref_image
    curr_perturbation = 0
    for iter in range(nb_it):
        curr_pertubation = generate_perturbation(size, eps, norm)
        curr_perturbated_img = ref_image + curr_perturbation
        prev_perturbated_img = ref_image + prev_perturbation
        curr_loss = K.eval(loss()
        # compute the loss of previous pert and current pert
        # deduces the gradient of the loss
        # compute the new pert and updates the previous pert
    return curr_pertubation

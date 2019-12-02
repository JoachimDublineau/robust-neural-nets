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

# # Test generate_perturbation
# perturbation = generate_perturbation((10,10), 1, norm = "l2")
# print(perturbation)
# print(np.linalg.norm(perturbation))

def compute_grad(func, point, delta):
    func = np.reshape(func, -1)
    for 

def maximize_loss(model, loss, ref_image, rate, eps, norm = "l2", 
                  nb_it = 10):
    size = ref_image.shape
    prev_perturbation  = np.zeros(size, dtype = int)
    prev_perturbated_img = ref_image
    prev_prediction = model.predict([prev_perturbated_img])
    prev_loss = 0
    print(prev_prediction.shape)
    curr_perturbation = generate_perturbation(size, eps, norm)
    for iter in range(nb_it):
        curr_perturbated_img = ref_image + curr_perturbation
        # prev_perturbated_img = ref_image + prev_perturbation
        curr_prediction = model.predict([curr_perturbated_img])
        curr_loss = K.eval(loss(curr_prediction, prev_prediction))
        grad = (curr_loss - prev_loss)/np.linalg.norm(curr_pertubation /
            - prev_perturbation)
        prev_perturbation = curr_perturbation
        curr_perturbation += rate *  
        
        # compute the loss of previous pert and current pert
        # deduces the gradient of the loss
        # compute the new pert and updates the previous pert
    return curr_pertubation

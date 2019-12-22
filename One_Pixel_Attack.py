import numpy as np
from tensorflow.keras import backend as K

def generate_one_pixel_attacks_on_batch(model, images, labels):
    """
    Computes a one pixel attack for a given x, model and loss.
    INPUTS:
    - model: tensorflow.keras model compiled with the loss and having
    input shape = images.shape.
    - images: array representing the input image.
    - labels: label of the input image.
    OUTPUTS:
    - attacks: array of the same shape as images representing the attacks.
    COMPUTATION TIME:
    Fast.
    """
    n,i,j,k = images.shape
    x_index = np.round(np.random.normal(i/2, int(6*i/32), n))
    y_index = np.round(np.random.normal(j/2, int(6*j/32), n))
    # z_index = np.random.randint(0, 3, n)
    attacks = []
    for p in range(n):
        index = int(x_index[p])
        x = (index if index>=0 and index<i else i-1)
        index = int(y_index[p])
        y = (index if index>=0 and index<j else j-1)
        image = images[p]
        image[x,y,:] *= eps
        attacks.append(image)
    return attacks

def generate_one_pixel_attacks(model, x, y, batch_size = 10):
    """
    Computes a one pixel attack for a given x, model and loss.
    INPUTS:
    - model: tensorflow.keras model compiled with the loss and having
    input shape = x.shape.
    - x: array representing the input image.
    - y: label of the input image.
    - batch_size: int, size of the batch for computation.
    OUTPUTS:
    - attacks: array of the same shape as x representing the attacks.
    COMPUTATION TIME:
    Fast.
    """
    predictions = np.argmax(model(K.cast(x,dtype = 'float32')), axis=1)
    ref = np.argmax(y, axis = 1)
    errors = ref - predictions
    images = []
    labels = []
    indexes = []
    for i,error in enumerate(errors):
        if error == 0:
            images.append(x[i])
            labels.append(y[i])
        else:
            indexes.append(i)
    images = np.array(images)
    labels = np.array(labels)
    tab_perturbations = []
    nb_batch = len(images) // batch_size + 1
    for i in range(nb_batch):
        end = 0
        if i == nb_batch - 1:
            end = len(images)
        else:
            end = (i+1)*batch_size
        x_batch = images[i*batch_size:end]
        y_batch = labels[i*batch_size:end]
        size = end - i*batch_size
        if size == 0:
            break
        print("Computing Attacks for batch", i, " Images", i*batch_size, "to", end)
        perturbations = generate_one_pixel_attacks_on_batch(model, x_batch, y_batch)
        for perturbation in perturbations:
            tab_perturbations.append(perturbation)
    attacks = []
    j = 0
    for i in range(x.shape[0]):
        if i in indexes:
            attacks.append(x[i])
        else:
            attacks.append(tab_perturbations[j])
            j+=1
    attacks = np.array(attacks, dtype = np.float32)
    return attacks

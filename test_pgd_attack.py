from PGD_attack import *
import src
import tensorflow as tf
import numpy as np
import time 
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K 

# # Test generate_perturbation
perturbation = generate_perturbation((10,10), 1)
print(perturbation)
print(np.linalg.norm(perturbation))

# Test projection:
a = np.array([2,2], dtype = np.float32)
b = np.array([1,1], dtype = np.float32)
print(a-b)
print(np.linalg.norm(a-b, ord=2))
print(projection(a, b, 1))

# # Test PGD Attack:
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
                             image, label, eps =1)
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

# # Test PGD Attack on batch:
x_train, y_train, x_test, y_test = src.cifar10.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

y_train = tf.keras.utils.to_categorical(y_train, \
    num_classes = len(src.cifar10.labels))
y_test = tf.keras.utils.to_categorical(y_test, \
    num_classes = len(src.cifar10.labels))

batch_size = 10
images = x_train[:batch_size]
labels = y_train[:batch_size]
perturbations = generate_pgd_attack_on_batch(model, categorical_crossentropy, 
                             images, labels, eps = 1, batch_size=batch_size)
print()
for i in range(batch_size):
    print("For image n°", i)
    print("Perturbation:")
    # print(perturbations[i])
    print("Norm of pertubation:", np.linalg.norm(perturbations[i]))
    print("Image:")
    # print(images[i])
    print("Norm:", np.linalg.norm(images[i]))
    print("Model prediction:", np.argmax(model(K.cast([images[i]], 
                                           dtype = 'float32'))[0]))
    print("Perturbated image:")
    print("Norm:", np.linalg.norm(images[i] + perturbations[i]))
    print("Model prediction:", np.argmax(model(K.cast([images[i] + perturbations[i]], 
                                        dtype = 'float32'))[0]))
    print()


# Test PGD Attack on batch:
x_train, y_train, x_test, y_test = src.cifar10.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

y_train = tf.keras.utils.to_categorical(y_train, \
    num_classes = len(src.cifar10.labels))
y_test = tf.keras.utils.to_categorical(y_test, \
    num_classes = len(src.cifar10.labels))

batch_size = 10
random_indexes = np.random.choice(x_test.shape[0], 100)

images = x_train[random_indexes]
labels = y_train[random_indexes]
print(images.shape)
print(labels.shape)
t0 = time.time()
perturbations = generate_pgd_attacks(model, categorical_crossentropy, 
                             images, labels, eps=1, batch_size=batch_size, step= 0.1,
                             threshold=1e-3, nb_it_max=20, accelerated=True)
t1 = time.time()
print("Computation time for 100 images:", t1-t0)
perturbations = generate_pgd_attacks(model, categorical_crossentropy, 
                             images, labels, eps=1, batch_size=batch_size, step= 0.1,
                             threshold=1e-3, nb_it_max=20, accelerated=False)
t2 = time.time()
print("Computation time for 100 images:", t2-t1)
print(perturbations[1])

def compute_efficiency(model, loss, x, y, eps, batch_size):
    attacks, perturbations, images, labels = generate_pgd_attacks_for_test(model, loss, x, y, eps, batch_size)
    predictions_on_pert = np.argmax(model(K.cast(images + perturbations, 
                                        dtype = 'float32')), axis=1)
    predictions = np.argmax(model(K.cast(images,dtype = 'float32')), axis=1)
    nb_mistakes = np.count_nonzero(predictions_on_pert-predictions)
    damages = nb_mistakes/images.shape[0]
    return damages
tab = []
tab_eps = []
for i in range(10):
    eps = 0.1 + i*0.2
    tab_eps.append(eps)
    tab.append(compute_efficiency(model, categorical_crossentropy, images, labels, eps, batch_size))
plt.xlabel("Eps")
plt.ylabel("Attack efficiency")
plt.plot(tab_eps, tab)
plt.show()
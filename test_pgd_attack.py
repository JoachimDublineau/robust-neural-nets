from PGD_attack import *
import src
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K

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
fig=plt.figure(figsize=(1, 2))
fig.add_subplot(1, 3, 1)
fig.suptitle('Image source vs Image attaquée (eps=1)', fontsize=16)
plt.imshow(image)
fig.add_subplot(1, 3, 2)
plt.imshow(image + perturbation)
plt.show()

# Test PGD Attack on batch:
random_indexes = np.random.choice(x_test.shape[0], 150)
batch_size = 50

images = x_test[random_indexes]
labels = y_test[random_indexes]
model.evaluate(images, labels)
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


def compute_accuracy(model, loss, x, y, eps, batch_size, norm = None):
    attacks = generate_pgd_attacks(model, loss, x, y, eps, batch_size, norm=norm)
    scores = model.evaluate(x + attacks, y)
    return scores[1]

tab = []
tab_eps = []
for i in range(11):
    eps = i*0.02
    tab_eps.append(eps)
    tab.append(compute_accuracy(model, categorical_crossentropy, images, labels, eps, batch_size,norm=np.inf))
plt.xlabel("Eps")
plt.ylabel("Accuracy")
plt.plot(tab_eps, tab)
plt.show()

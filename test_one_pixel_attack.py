import src
import tensorflow as tf
import numpy as np
import time 
import matplotlib.pyplot as plt
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K 
from One_Pixel_Attack import *

x_train, y_train, x_test, y_test = src.cifar10.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

y_train = tf.keras.utils.to_categorical(y_train, \
    num_classes = len(src.cifar10.labels))
y_test = tf.keras.utils.to_categorical(y_test, \
    num_classes = len(src.cifar10.labels))

model = tf.keras.models.load_model("models/cifar10_simple_model_73_acc.h5")
batch_size = 50
random_indexes = np.random.choice(x_test.shape[0], 500)
images = x_test[random_indexes]
labels = y_test[random_indexes]
attacks = generate_one_pixel_attacks(model, images, labels, batch_size)

fig=plt.figure(figsize=(1, 2))
fig.add_subplot(1, 2, 1)
plt.imshow(images[0])
fig.add_subplot(1, 2, 2)
plt.imshow(attacks[0])
plt.show()

fig=plt.figure(figsize=(1, 2))
fig.add_subplot(1, 2, 1)
plt.imshow(images[1])
fig.add_subplot(1, 2, 2)
plt.imshow(attacks[1])
plt.show()

fig=plt.figure(figsize=(1, 2))
fig.add_subplot(1, 2, 1)
plt.imshow(images[2])
fig.add_subplot(1, 2, 2)
plt.imshow(attacks[2])
plt.show()

def compute_accuracy(model, x, y, batch_size):
    attacks = generate_one_pixel_attacks(model, x, y, batch_size)
    scores = model.evaluate(attacks, y)
    return scores[1]

print(model.evaluate(images, labels)[1])
print(compute_accuracy(model, images, labels, batch_size))


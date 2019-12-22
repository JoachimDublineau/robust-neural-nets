In the script `cifar10_simple_network.py`, there is two methods to train the network. One uses the classical `fit` function from TensorFlow Keras, and the other one uses a custom training.

To attest the efficency and the correctness of the custom method, a simple CIFAR10 training on original pictures has been done using both methods.

The commands used are the following:
    python cifar10_simple_network.py -e 30 -b 128 -v # Simple method
    python cifar10_simple_network.py -e 30 -b 128 -v --train-method defense-fgsm --alpha 1 --epsilon 0 --epsilon-growth 0 # Custom method

We can see by the results, that the evolution of loss and accuracy are sensibly identical.

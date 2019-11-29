# Hex - Goldorak-42
This is the final assignment of the Data Science Project course of the [IASD Master](https://www.lamsade.dauphine.fr/wp/iasd/en/) (Artificial Intelligence Systems and Data Science). The goal is to code two adversarial attacks, FGSM and PGD, and see that a simple neural network is not robust against small adversarial perturbations.

## Getting Started
### Prerequisites
You need to have Python 3 > 3.6 installed on your system.

### Installing
* 1) To install required packages, you can run the following command.

    pip install -r requirements.txt

* 2) Clone the repository.

    git clone https://github.com/Dauphine-Java-M1/robust-neural-nets-476f6c646f72616b-42.git

* 3) You are good to go!

## Usage
### `cifar10_simple_network.py`
This script allows you to train a simple network on the CIFAR10 dataset.

For example, to train a network on 50 epochs, with a batch size of 32, and to save the weights in "weights.h5" file, you can run the following command.

    python cifar10_simple_network.py --epochs 50 --batch-size 32 --weights weights.h5 --verbose

To get more preicison, you can run the following command.

    `python cifar10_simple_network.py -h`

## Authors
* Joachim Dublineau
* Elie Kadoche
* Thomas Petiteau

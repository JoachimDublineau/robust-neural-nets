# Hex - Goldorak-42
This is the final assignment of the Data Science Project course of the [IASD Master](https://www.lamsade.dauphine.fr/wp/iasd/en/) (Artificial Intelligence Systems and Data Science). The goal is to code two adversarial attacks, FGSM and PGD, and see that a simple neural network is not robust against small adversarial perturbations.

## Getting Started
### Prerequisites
* Python 3

You need Python 3.6 to Python 3.7 to run this project. An install guide can be found [here](https://wiki.python.org/moin/BeginnersGuide/Download).

### Installing
* 1) Clone the repository.

    git clone https://github.com/Dauphine-Java-M1/robust-neural-nets-476f6c646f72616b-42.git

* 2) Install python requirements with following command.

    pip install -r requirements.txt

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

# Hex - Goldorak-42

This is the final assignment of the Data Science Project course of the [IASD Master](https://www.lamsade.dauphine.fr/wp/iasd/en/) (Artificial Intelligence Systems and Data Science). The goal is to code two adversarial attacks, FGSM and PGD, and see that a simple neural network is not robust against small adversarial perturbations.

## Getting Started

### Prerequisites

* Python 3

You need Python 3.6 to Python 3.7 to run this project. An install guide can be found [here](https://wiki.python.org/moin/BeginnersGuide/Download).

### Installing

1) Clone the repository.

```
git clone <https://github.com/Dauphine-Java-M1/robust-neural-nets-476f6c646f72616b-42.git>
```

2) Install python requirements with following command.

```
pip install -r requirements.txt
```

## Usage

### `cifar10_simple_network.py`

This script allows you to train a simple network on the CIFAR10 dataset.

For example, to train a network on 50 epochs, with a batch size of 32, and to save the weights in "weights.h5" file, you can run the following command.

```
python cifar10_simple_network.py --epochs 50 --batch-size 32 --weights weights.h5 --verbose
```

This script also allows you to train a robust network against FGSM attacks on the CIFAR10 dataset.

For example, to train such a network with a first epsilon of 0 (incrementing by 0.005) and with an alpha of 0.5, you can run the following command.

```
python cifar10_simple_network.py --epochs 50 --batch-size 32 --train-method defense-fsgm --epsilon 0 --epsilon-growth 0.005 --alpha 0.5
```

To get more info, run the following command.

```
python cifar10_simple_network.py -h
```

### `train_log_to_graphs.py`

This script creates a graph based on the output of `cifar10_simple_network.py` script.

To get more info, run the following command.

```
python train_log_to_graphs.py -h
```

### `fgsm.py`

This script attacks a given cifar10 network using FGSM ax explained in [this](https://arxiv.org/abs/1412.6572) article.

To get more info, run the following command.

```
python fgsm.py -h
```

### `fgsm_log_to_graphs.py`

This script creates a graph based on the output of `fgsm.py` script.

To get more info, run the following command.

```
python fgsm_log_to_graphs.py
```

### `PGD_attack.py`

This script contains the required tools to create PGD attacks on a single image as well as on a set of images.

The most important function is generate_pgd_attacks which can be executed with the following command.

```
perturbations = generate_pgd_attacks(
    model,
    categorical_crossentropy,
    images,
    labels,
    eps=1,
    batch_size=batch_size,
    step= 0.1,
    threshold=1e-3,
    nb_it_max=20,
    accelerated=False,
)
```

## Authors

* Joachim Dublineau
* Elie Kadoche
* Thomas Petiteau

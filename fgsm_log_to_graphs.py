import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd

import src

parser = argparse.ArgumentParser(
    description="Script to create graphs from fgsm logs")
parser.add_argument("path", help="path to the fgsm log file")
parser.add_argument(
    "-o",
    "--output",
    help="Path of the ouput file. Default is {}fgsm_graphs.png".format(
        src.results_dir
    ),
    type=str,
    default="{}fgsm_graphs.png".format(src.results_dir),
)

args = parser.parse_args()
path = args.path
output = args.output

src.create_dir_if_not_found(src.results_dir)

# Reading data
# -------------------------

print("Reading data...", end="")

data = pd.read_csv(path, sep=";")
epsilon = 'epsilon'

print("Done!")

# Saving graphs
# -------------------------

print("Creating graphs...", end="")

data.sort_values(by=['epsilon'], inplace=True)

fig, axes = plt.subplots(1, data.shape[1] - 1)

for metric_index, metric in enumerate(data.columns[1:]):
    if metric != epsilon:
        axes[metric_index].plot(data[epsilon], data[metric])
        axes[metric_index].set_ylabel(metric)
        axes[metric_index].set_xlabel(epsilon)

print("Done!")

print("Saving graphs...", end="")

fig.tight_layout()
fig.savefig(output, dpi=400, transparent=True)

print("Done!")

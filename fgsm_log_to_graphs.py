import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd

import src

parser = argparse.ArgumentParser(
    description="Script to create graphs from fgsm logs")
parser.add_argument("path", help="path to the fgsm logs files. It is assumed that the files have the same headers", nargs='+')
parser.add_argument(
    "-o",
    "--output",
    help="Path of the ouput file. Default is {}fgsm_graphs.png".format(
        src.results_dir
    ),
    type=str,
    default="{}fgsm_graphs.png".format(src.results_dir),
)
parser.add_argument('--labels', help='the label associate to each files', nargs='+')

args = parser.parse_args()
output = args.output

src.create_dir_if_not_found(src.results_dir)

data = pd.read_csv(args.path[0], sep=';')

fig, axes = plt.subplots(1, data.shape[1] - 1)

# Reading data
# -------------------------

for index in  range(len(args.path)):
    path = args.path[index]
    print("Reading data...", end="")

    data = pd.read_csv(path, sep=";")
    epsilon = 'epsilon'

    print("Done!")

    # Saving graphs
    # -------------------------

    print("Creating graphs...", end="")

    data.sort_values(by=['epsilon'], inplace=True)


    for metric_index, metric in enumerate(data.columns[1:]):
        if metric != epsilon:
            axes[metric_index].plot(data[epsilon], data[metric], label=None if args.labels is None else args.labels[index], alpha=0.7 if args.labels is not None else 1.0)
            axes[metric_index].set_ylabel(metric)
            axes[metric_index].set_xlabel(epsilon)

    print("Done!")

print("Saving graphs...", end="")

if args.labels is not None:
    for ax in axes:
        ax.legend()
fig.tight_layout()
fig.savefig(output, dpi=400, transparent=True)

print("Done!")

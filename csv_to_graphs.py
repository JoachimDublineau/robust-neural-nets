import argparse

import matplotlib.pyplot as plt
import pandas as pd

import src

parser = argparse.ArgumentParser(description="Scipt to create graphs from training log")
parser.add_argument("path", help="The name of the training log file")
parser.add_argument(
    "-o",
    "--output",
    help="Path of the ouput file. Default is {}training_graphs.png".format(
        src.results_dir
    ),
    type=str,
    default="{}training_graphs.png".format(src.results_dir),
)

args = parser.parse_args()
path = args.path
output = args.output

src.create_dir_if_not_found(src.results_dir)

# Reading data
# -------------------------

print("Reading data...")

data = pd.read_csv(path, sep=";")

loss = data["loss"]
val_loss = data["val_loss"]

accuracy = data["accuracy"]
val_accuracy = data["val_accuracy"]

print("Data is loaded.")

# Saving graphs
# -------------------------

print("Saving graphs...")

fig, ax = plt.subplots(nrows=1, ncols=2, num="Policy head", figsize=(11, 4))
ax[0].plot(loss, color="c")
ax[0].plot(val_loss, color="r")
ax[0].title.set_text("Loss (categorical crossentropy)")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].legend(["Train", "Valid"], loc="upper right")

ax[1].plot(accuracy, color="c")
ax[1].plot(val_accuracy, color="r")
ax[1].title.set_text("Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend(["Train", "Valid"], loc="upper right")

fig.savefig(output, dpi=400, transparent=True)

print("Graphs are saved.")

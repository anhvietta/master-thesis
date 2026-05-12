"""
Compute the precision/recall distribution of number of aligned residues.
Parameter:
- diag_data_path: Data generated using the ss_vs_sw_flat.py
"""

import matplotlib.pyplot as plt
import numpy as np

diag_data_path = "../diag_data"
data = []
with open(diag_data_path, 'r') as handler:
    for line in handler:
        split = line.strip().split("\t")
        if len(split) != 4:
            break
        data.append(split)
precision = np.array([float(d[2]) for d in data])
recall = np.array([float(d[3]) for d in data])
numbins = 20
for i, attr in enumerate([precision, recall]):
    # --- Histogram ---
    counts, bin_edges = np.histogram(attr, bins=numbins)

    print("Histogram counts:", counts)
    print("Bin edges:", bin_edges)

    # Optional: visualize histogram
    fig, ax = plt.subplots()
    ax.hist(attr, bins=numbins, edgecolor='black')
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    if i == 0:
        ax.set_title("Histogram of precision values of diagonals")
        plt.savefig("../diag_precision.png")
    else:
        ax.set_title("Histogram of recall values of diagonals")
        plt.savefig("../diag_recall.png")

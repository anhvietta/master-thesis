"""
Make a histogram for diagonal recovery data
Parameter:
- diag_hist_data: Data generated using the ss_vs_sw_flat.py
"""

import numpy as np
import matplotlib.pyplot as plt
diag_hist_data = "/scratch/stud2018/ata/measurements2/diag_hist_data"
with open(diag_hist_data, "r") as handler:
    lines = list(handler)
    sw_data = np.array([int(c)
                       for c in lines[0][1:-2].split(',')], dtype=np.uint16)
    ss_data = np.array([int(c)
                       for c in lines[1][1:-2].split(',')], dtype=np.uint16)

# Define common bins
bins = np.linspace(min(sw_data), max(sw_data), 30)

# Plot histograms
plt.figure(figsize=(8, 5))

plt.hist(sw_data, bins=bins, alpha=0.5, label='SW Diags', edgecolor='black')
plt.hist(ss_data, bins=bins, alpha=0.7, label='SS Diags', edgecolor='black')

# Labels and legend
plt.xlabel('Diag length')
plt.ylabel('Log Count')
plt.yscale('log')
plt.title('Histogram of diagonals found vs reference depending on length')
plt.legend()

plt.tight_layout()
plt.savefig("/scratch/stud2018/ata/measurements2/diags.png")

# Plot histograms
plt.figure(figsize=(8, 5))

counts_full, _ = np.histogram(sw_data, bins=bins)
counts_subset, _ = np.histogram(ss_data, bins=bins)

# Bin centers for plotting
bin_centers = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)

# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    pct = counts_subset / counts_full * 100
    pct[counts_full == 0] = 0

plt.bar(bin_centers, pct, width=width, edgecolor='black')

plt.xlabel('Diag length')
plt.ylabel('Recovery rate')
plt.title('Histogram of diagonals recovery rate depending on length')
plt.legend()

plt.tight_layout()
plt.savefig("/scratch/stud2018/ata/measurements2/recovery.png")

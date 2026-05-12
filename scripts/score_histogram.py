"""
Generate cutoff references for SW measurements
"""

import numpy as np
import matplotlib.pyplot as plt

sw_file = "/scratch/stud2018/ata/ma_data/swref/0.txt"
with open(sw_file, 'r') as handler:
    sw_data = [line.strip().split("\t") for line in list(handler)]

# Example unsorted float data
data = [d[7] for d in sw_data]

# Convert to NumPy array
arr = np.array(data, dtype=np.float32)

# --- Histogram ---
counts, bin_edges = np.histogram(arr, bins=10)

print("Histogram counts:", counts)
print("Bin edges:", bin_edges)

# Optional: visualize histogram
plt.hist(arr, bins=10, edgecolor='black')
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.savefig("gttl_score_histogram.png")

# --- Percentile cutoffs (every 10%) ---
percentiles = np.arange(0, 110, 10)  # 0%, 10%, ..., 100%
cutoffs = np.percentile(arr, percentiles)

print("\nPercentile cutoffs:")
for p, c in zip(percentiles, cutoffs):
    print(f"{p}% -> {c:.3f}")

outdir = "/scratch/stud2018/ata/ma_data/swref/"
for c in cutoffs:
    data = ["\t".join([str(a) for a in d]) for d in sw_data if float(d[7]) > c]
    with open(outdir + f"cutoff_{c:.1f}.txt", "w") as handler:
        handler.write("\n".join(data))

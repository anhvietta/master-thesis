"""
Plots the comparison between ProtSearchFlat and ProtSearchIVFPQ.
Parameter:
- cache_path: path to cached measurements from benchmark_pr.py and benchmark_pr_ivfpq.py
"""

import matplotlib.pyplot as plt
import re
import numpy as np


def parse_file(file_path):
    steps_time = []
    whole_time = []
    with open(file_path, 'r') as handler:
        for line in handler:
            if len(steps_time) >= 2 and len(whole_time) >= 6:
                break
            if line.startswith("#Time:\t"):
                steps_time.append(
                    float(re.search(r"(-?\d+\.?\d*?)", line).group(1)))
            elif line.startswith("# TIME\t"):
                whole_time.append(
                    float(re.search(r"(-?\d+\.?\d*?)", line).group(1)))
    data = {
        "Import Index": whole_time[0],
        "Encode Sequences": steps_time[0],
        "Query Sequences": steps_time[1],
        "Assemble Alignments": sum(whole_time[2:])
    }
    return data


def plot(data, labels, ax):
    sample_key = list(data.keys())[0]
    bottom = np.zeros(len(data[sample_key]))
    for l, d in data.items():
        ax.bar([l.replace('_', ",")
               for l in labels], d, 0.75, label=l, bottom=bottom)
        bottom += d
    for l, b in zip(labels, bottom):
        plt.text(l.replace('_', ","), b, str(b), ha="center",
                 va="bottom", fontsize=8, fontweight="bold")
    # ax.set_xticklabels(labels)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Run time (ms)")
    ax.set_title(
        "Processing time comparison of various flat and IVFPQ configs")


cache_path = "../measurements2/cache"
configs = ["8_0.700", "16_0.700", "32_0.700", "64_0.700", "128_0.600"]
data = {}
labels = []
for c in configs:
    cs = c + "_3_2"
    ivfpq_cs = cs + "_32"
    flat_data = parse_file(cache_path + "/out_" + cs + ".tsv")
    ivfpq_data = parse_file(
        cache_path + "/out_ivfpq_8192_" + ivfpq_cs + ".tsv")
    # data += [flat_data , ivfpq_data]
    labels += [c, c + "_32"]
    for d in [flat_data, ivfpq_data]:
        for key in d:
            if key not in data:
                data[key] = []
            data[key].append(d[key])

fig, ax = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(12)
plot(data, labels, ax)
ax.set_ylim(0, 400000)
ax.legend()
plt.savefig("/scratch/stud2018/ata/measurements2/flat_vs_ivfpq.png")

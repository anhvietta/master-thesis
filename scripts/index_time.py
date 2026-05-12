"""
Plot indexing (or querying) time for various config of ProtSearchIVFPQ
Parameter:
- input_path: Path to files generated manually by running ProtSearch against some database size (by default 100k, 200k, 500k, 1m)
"""

from pathlib import Path
import re
import matplotlib.pyplot as plt
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
    '''data = {
        "Import Index": whole_time[0],
        "Encode Sequences": steps_time[0],
        "Query Sequences": steps_time[1],
        "Assemble Alignments": sum(whole_time[2:])
    }'''
    data = {
        "Build index": whole_time[1]/1000
    }
    return data


def plot(data, labels, ax):
    print(labels)
    sample_key = list(data.keys())[0]
    bottom = np.zeros(len(data[sample_key]))
    for l, d in data.items():
        ax.bar(labels, d, 0.5, label=l, bottom=bottom)
        bottom += d
    # ax.set_xticklabels(labels)
    ax.set_xlabel("Database size")
    ax.set_ylabel("Run time (s)")
    ax.set_title(
        "Index time profile using different database size, $k=128,s_{min}=0.6,n_{probe}=32$")


input_path = "../db_out_"

fig, ax = plt.subplots()
fig.set_figwidth(12)
fig.set_figheight(6)
suffix = {"100 000": "100k", "200 000": "200k",
          "500 000": "500k", "1 000 000": "1m"}
files = {k: input_path + v for k, v in suffix.items()}
data = {}
labels = []
for i, (c, file) in enumerate(files.items()):
    d = parse_file(file)
    for key in d:
        if key not in data:
            data[key] = []
        data[key].append(d[key])
    labels.append(c)
plot(data, labels, ax)
plt.legend()
plt.tight_layout()
plt.savefig("../measurements2/index_db.png")

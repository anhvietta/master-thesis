"""
Plot training statistics
"""

import matplotlib.pyplot as plt
import csv

base_path = "/scratch/stud2018/ata/measurements2/"
match_train_file = base_path + \
    "0324-1328_500_0.0004_0.07_32+96+160+256+320+384+480_3+5+7+9+11+15+17_1+2+3+4+5+6+7_320_traintotal.csv"
match_test_file = base_path + \
    "0324-1328_500_0.0004_0.07_32+96+160+256+320+384+480_3+5+7+9+11+15+17_1+2+3+4+5+6+7_320_testtotal.csv"
'''
with open(match_train_file, "r") as handler:
    data = list(csv.DictReader(handler))
    steps = [int(d["Step"]) for d in data]
    value = [float(d["Value"]) for d in data]

with open(match_test_file, "r") as handler:
    data = list(csv.DictReader(handler))
    tsteps = [int(d["Step"]) for d in data]
    tvalue = [float(d["Value"]) for d in data]

fig, ax = plt.subplots()
ax.plot(steps, value, label="Training")
ax.scatter(tsteps, tvalue, label="Validation")
for s in tsteps:
    ax.axvline(x=s, linestyle=":", color="gray", alpha=0.7)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_xlabel("Steps")
ax.set_ylabel("Value")
ax.set_title("Total")
plt.legend()
plt.savefig(base_path + "total_train.png")
'''

token_file = base_path + \
    "0324-1328_500_0.0004_0.07_32+96+160+256+320+384+480_3+5+7+9+11+15+17_1+2+3+4+5+6+7_320_tokenizer.csv"
with open(token_file, "r") as handler:
    data = list(csv.DictReader(handler))
    steps = [int(d["Step"]) for d in data]
    value = [float(d["Value"]) for d in data]

fig, ax = plt.subplots()
ax.plot(steps, value, label="Training")
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_xlabel("Steps")
ax.set_ylabel("Value")
ax.set_title("Graph Laplacian")
plt.legend()
plt.savefig(base_path + "tokenizer_plot.png")

"""
Plot PR curves
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import re
from adjustText import adjust_text


def prepare_pr_curve(recall, precision):
    # Convert to numpy arrays
    recall = np.asarray(recall)
    precision = np.asarray(precision)

    # 1. Sort by recall
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]

    # 2. Deduplicate recall values (keep max precision)
    unique_recalls = np.unique(recall)
    max_precision = []
    for r in unique_recalls:
        max_precision.append(np.max(precision[recall == r]))
    recall = unique_recalls
    precision = np.array(max_precision)

    # 3. Enforce monotonic precision (precision envelope)
    # traverse from right to left
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # 4. Add boundary point (0,1)
    if recall[0] > 0:
        recall = np.insert(recall, 0, 0.0)
        precision = np.insert(precision, 0, 1.0)

    return recall, precision


def prepare_pr_curve_with_indices(recall, precision):
    recall = np.asarray(recall)
    precision = np.asarray(precision)

    orig_idx = np.arange(len(recall))

    # 1. Sort by recall
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]
    orig_idx = orig_idx[order]

    # 2. Deduplicate recall values (keep max precision)
    unique_recalls = np.unique(recall)
    new_precision = []
    new_indices = []

    for r in unique_recalls:
        mask = (recall == r)
        idxs = np.where(mask)[0]
        best_local = idxs[np.argmax(precision[mask])]

        new_precision.append(precision[best_local])
        new_indices.append(orig_idx[best_local])

    recall = unique_recalls
    precision = np.array(new_precision)
    orig_idx = np.array(new_indices)

    # 3. Precision envelope (right-to-left)
    for i in range(len(precision) - 2, -1, -1):
        if precision[i] < precision[i + 1]:
            precision[i] = precision[i + 1]
            orig_idx[i] = orig_idx[i + 1]

    # 4. Remove redundant horizontal points (keep rightmost)
    keep = [True] * len(recall)
    for i in range(len(recall) - 1):
        if precision[i] == precision[i + 1]:
            keep[i] = False  # drop left one

    recall = recall[keep]
    precision = precision[keep]
    orig_idx = orig_idx[keep]

    # 5. Add boundary (0,1)
    if recall[0] > 0:
        recall = np.insert(recall, 0, 0.0)
        precision = np.insert(precision, 0, 1.0)
        orig_idx = np.insert(orig_idx, 0, -1)

    return recall, precision, orig_idx


def compute_aucpr(recall, precision):
    # Step-wise AUCPR
    auc = 0.0
    for i in range(1, len(recall)):
        auc += (recall[i] - recall[i-1]) * precision[i]
    return auc


def plot_pr_curve(recall, precision, raw_data, label="ProtSearch", annotate=False):
    print(raw_data)
    recall, precision, idx = prepare_pr_curve_with_indices(recall, precision)
    auc = compute_aucpr(recall, precision)

    plt.step(recall, precision, where='pre',
             label=label + f' AUCPR = {auc:.4f}')
    # plt.scatter(recall, precision)
    if annotate:
        texts = []
        for r, p, i in list(zip(recall, precision, idx)):
            if i >= 0:
                d = raw_data[i]
                if len(raw_data[0]) < 9:
                    label = plt.text(
                        r, p, f"{int(d[1])},{float(d[2]):.1f}", fontsize=8)
                else:
                    label = plt.text(
                        r, p, f"{int(d[1])},{float(d[2]):.1f},{int(d[5])}", fontsize=8)
                # plt.annotate(label, (r, p), textcoords="offset points", xytext=(3,3), fontsize=8)
                texts.append(label)
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5)
        )
    return auc


def plot_pr_curve_mmseqs(recall, precision, sensitivity):
    recall, precision, idx = prepare_pr_curve_with_indices(recall, precision)
    auc = compute_aucpr(recall, precision)

    plt.step(recall, precision, where='pre',
             label=f'MMseqs2 AUCPR = {auc:.4f}')
    texts = []
    for r, p, i in list(zip(recall, precision, idx)):
        if i >= 0:
            d = sensitivity[i]
            label = plt.text(r, p, f"{float(d):.1f}", fontsize=8)
            # plt.annotate(label, (r, p), textcoords="offset points", xytext=(3,3), fontsize=8)
            texts.append(label)
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5)
    )
    return auc


def plot_pr_curve_diamond(recall, precision, sensitivity):
    recall, precision, idx = prepare_pr_curve_with_indices(recall, precision)
    auc = compute_aucpr(recall, precision)

    plt.step(recall, precision, where='pre',
             label=f'DIAMOND AUCPR = {auc:.4f}')
    texts = []
    for r, p, i in list(zip(recall, precision, idx)):
        if i >= 0:
            d = sensitivity[i]
            label = plt.text(r, p, d, fontsize=8)
            # plt.annotate(label, (r, p), textcoords="offset points", xytext=(3,3), fontsize=8)
            texts.append(label)
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5)
    )
    return auc


plt.figure(figsize=(16, 8))
plt.xlim(0., 1.)
with open("/scratch/stud2018/ata/measurements2/pr_flat.tsv", "r") as handler:
    reader = csv.reader(handler, delimiter="\t")
    raw_data = [row for row in reader if row[0] ==
                "47.0" and row[4] == "2" and row[3] == "3"]
recall = [float(p[-3]) for p in raw_data]
precision = [float(p[-2]) for p in raw_data]
auc = plot_pr_curve(recall, precision, raw_data,
                    label="ProtSearchFlat", annotate=True)
print("AUCPR:", auc)

with open("/scratch/stud2018/ata/measurements2/ivfpq_pr.tsv", "r") as handler:
    reader = csv.reader(handler, delimiter="\t")
    raw_data = [row for row in reader if row[0] ==
                "47.0" and row[3] == "3" and row[4] == "2" and row[5] == "32"]
print("*".join(["&".join(c for c in r) for r in raw_data]))
recall = [float(p[-3]) for p in raw_data]
precision = [float(p[-2]) for p in raw_data]
auc = plot_pr_curve(recall, precision, raw_data,
                    label="ProtSearchIVFPQ", annotate=True)
print("AUCPR:", auc)

mmseqs_datafiles = Path("/scratch/stud2018/ata/mmseqs_out/swref").glob("*.txt")
mmseqs_recall = []
mmseqs_precision = []
mmseqs_sensitivity = []
for f in mmseqs_datafiles:
    mmseqs_sensitivity.append(float(f.name[:-4]))
    with open(f.resolve(), 'r') as handler:
        lines = list(handler)
        recall = float(re.search(r"-?\d+(\.\d+)?", lines[0]).group(1))
        precision = float(re.search(r"-?\d+(\.\d+)?", lines[1]).group(1))
        mmseqs_recall.append(recall)
        mmseqs_precision.append(precision)
# print(mmseqs_recall, mmseqs_precision, mmseqs_sensitivity)
auc = plot_pr_curve_mmseqs(mmseqs_recall, mmseqs_precision, mmseqs_sensitivity)
print("AUCPR:", auc)

diamond_sensitivity = ["fast", "mid_sensitive", "sensitive",
                       "more_sensitive", "very_sensitive", "ultra_sensitive"]
diamond_datafiles = [("/scratch/stud2018/ata/diamond_out/" + s + "/pr")
                     for s in diamond_sensitivity]
diamond_recall = []
diamond_precision = []
for f in diamond_datafiles:
    with open(f, 'r') as handler:
        lines = list(handler)
        recall = float(re.search(r"-?\d+(\.\d+)?", lines[0]).group(1))
        precision = float(re.search(r"-?\d+(\.\d+)?", lines[1]).group(1))
        diamond_recall.append(recall)
        diamond_precision.append(precision)
# print(mmseqs_recall, mmseqs_precision, mmseqs_sensitivity)
auc = plot_pr_curve_diamond(
    diamond_recall, diamond_precision, diamond_sensitivity)
print("AUCPR:", auc)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
# plt.savefig("/scratch/stud2018/ata/measurements2/pr_curve_flat.png")

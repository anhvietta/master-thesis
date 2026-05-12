"""
Compute some statistics to showcase the difference between false positive and true positive samples
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

ss_file = sys.argv[1]
sw_file = sys.argv[2]
sim_file = sys.argv[3]

with open(ss_file, 'r') as handler:
    ss_data = [line.strip().split("\t") for line in list(handler)[41:]]

with open(sw_file, 'r') as handler:
    sw_data = [line.strip().split("\t") for line in list(handler)]

with open(sim_file, 'r') as handler:
    sim_data = [line.strip().split("\t") for line in list(handler)[32:-5]]
    sim_dict = {}
    for e in sim_data:
        try:
            sim_dict[(e[0], e[1])][(int(e[2]), int(e[3]))] = float(e[4])
        except KeyError as err:
            sim_dict[(e[0], e[1])] = {}
            sim_dict[(e[0], e[1])][(int(e[2]), int(e[3]))] = float(e[4])


ss_seqpair_set = set((e[0], e[1]) for e in ss_data)
sw_seqpair_set = set((e[0], e[1]) for e in sw_data)

ss_dict = {(e[0], e[1]): e[-4:] for e in ss_data}

intersection = ss_seqpair_set.intersection(sw_seqpair_set)
false_positives = ss_seqpair_set.difference(intersection)

true_positives_data = {k: ss_dict[k] for k in intersection}
false_positives_data = {k: ss_dict[k] for k in false_positives}

avg_length_data = {
    "tp": [],
    "fp": [],
    "name": "Average alignment length"
}
avg_diff_data = {
    "tp": [],
    "fp": [],
    "name": "diff"
}
aligned_residue_data = {
    "tp": [],
    "fp": [],
    "name": "Number of aligned residues"
}
score_data = {
    "tp": [],
    "fp": [],
    "name": "Score"
}
avg_sim_data = {
    "tp": [],
    "fp": [],
    "name": "Average similarity"
}
max_sim_data = {
    "tp": [],
    "fp": [],
    "name": "Max similarity"
}
tolerance = 20

tp_data = []
fp_data = []
for (t, q), m in true_positives_data.items():
    score = float(m[0])
    vt = np.array([c == "1" for c in m[2]], dtype=bool)
    vq = np.array([c == "1" for c in m[3]], dtype=bool)
    vt_pos = np.argwhere(vt)
    vq_pos = np.argwhere(vq)
    tlen = (vt_pos[-1] - vt_pos[0] + 1)[0]
    qlen = (vq_pos[-1] - vq_pos[0] + 1)[0]
    total_sim = 0
    max_sim = -1
    for tpos, qpos in zip(vt_pos, vq_pos):
        diag = (tpos - qpos)[0]
        try:
            s = sim_dict[(t, q)][(int(diag), int(qpos[0]))]
        except:
            found = False
            for (dk, pk) in sim_dict[(t, q)].keys():
                if dk == diag and abs(pk - int(qpos[0])) <= tolerance:
                    found = True
                    s = sim_dict[(t, q)][(dk, pk)]
                    break
            if not found:
                for (dk, pk) in sim_dict[(t, q)].keys():
                    if dk == diag:
                        print((t, q, dk, pk))
                print(vq_pos)
                print(qpos[0])
                raise RuntimeError("Error")
        total_sim += s
        max_sim = max(max_sim, s)
    avg_length_data["tp"].append(float(qlen + tlen)/2)
    avg_diff_data["tp"].append(abs(qlen-tlen))
    score_data["tp"].append(score)
    aligned_residue_data["tp"].append(np.sum(vt))
    avg_sim_data["tp"].append(total_sim / np.sum(vt))
    max_sim_data["tp"].append(max_sim)
    tp_data.append([
        t,
        q,
        str(float(qlen + tlen)/2),
        str(abs(qlen-tlen)),
        str(score),
        str(np.sum(vt)),
        str(total_sim / np.sum(vt)),
        str(max_sim)
    ])


with open("/scratch/stud2018/ata/measurements2/tp_ivfpq.tsv", "w") as handler:
    handler.write("\n".join(["\t".join(d) for d in tp_data]))

for (t, q), m in false_positives_data.items():
    score = float(m[0])
    vt = np.array([c == "1" for c in m[2]], dtype=bool)
    vq = np.array([c == "1" for c in m[3]], dtype=bool)
    vt_pos = np.argwhere(vt)
    vq_pos = np.argwhere(vq)
    tlen = (vt_pos[-1] - vt_pos[0] + 1)[0]
    qlen = (vq_pos[-1] - vq_pos[0] + 1)[0]
    total_sim = 0
    max_sim = -1
    for tpos, qpos in zip(vt_pos, vq_pos):
        diag = (tpos - qpos)[0]
        try:
            s = sim_dict[(t, q)][(int(diag), int(qpos[0]))]
        except:
            found = False
            for (dk, pk) in sim_dict[(t, q)].keys():
                if dk == diag and abs(pk - int(qpos[0])) <= tolerance:
                    found = True
                    s = sim_dict[(t, q)][(dk, pk)]
                    break
            if not found:
                for (dk, pk) in sim_dict[(t, q)].keys():
                    if dk == diag:
                        print((t, q, dk, pk))
                print(vq_pos)
                print(qpos[0])
                raise RuntimeError("Error")
        total_sim += s
        max_sim = max(max_sim, s)
    avg_length_data["fp"].append(float(qlen + tlen)/2)
    avg_diff_data["fp"].append(abs(qlen-tlen))
    score_data["fp"].append(score)
    aligned_residue_data["fp"].append(np.sum(vt))
    avg_sim_data["fp"].append(total_sim / np.sum(vt))
    max_sim_data["fp"].append(max_sim)
    fp_data.append([
        t,
        q,
        str(float(qlen + tlen)/2),
        str(abs(qlen-tlen)),
        str(score),
        str(np.sum(vt)),
        str(total_sim / np.sum(vt)),
        str(max_sim)
    ])

with open("/scratch/stud2018/ata/measurements2/fp_ivfpq.tsv", "w") as handler:
    handler.write("\n".join(["\t".join(d) for d in fp_data]))


# Plot histograms
fig, ax = plt.subplots(5, 2)
fig.set_figheight(24)
fig.set_figwidth(16)
for i, data in enumerate([avg_length_data, score_data, aligned_residue_data, avg_sim_data, max_sim_data]):
    print(len(data["fp"]), len(data["tp"]))
    # Define common bins
    minval = np.min(data["fp"]) if np.min(data["fp"]) < np.min(
        data["tp"]) else np.min(data["tp"])
    maxval = np.max(data["fp"]) if np.max(data["fp"]) > np.max(
        data["tp"]) else np.max(data["tp"])
    bins = np.linspace(minval, maxval, 30)

    ax[i, 0].hist(data["fp"], bins=bins, label='fp', edgecolor='black')
    ax[i, 1].hist(data["tp"], bins=bins, label='tp', edgecolor='black')

    # Labels and legend
    ax[i, 0].set_xlabel('Feature')
    ax[i, 0].set_ylabel('Count')
    # ax[i, 0].set_yscale('log')
    ax[i, 0].set_title(data["name"] + " FP")
    ax[i, 0].legend()
    ax[i, 1].set_xlabel('Feature')
    ax[i, 1].set_ylabel('Count')
    # ax[i, 1].set_yscale('log')
    ax[i, 1].set_title(data["name"] + " TP")
    ax[i, 1].legend()

fig.tight_layout()
plt.savefig("/scratch/stud2018/ata/measurements2/fp_sim_0_0_normal.png")

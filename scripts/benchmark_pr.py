"""
This script creates some parameter sets using grid search and runs ProtSearch
using the sets and compute precision/recall/f1-score against reference
Smith-Waterman at various cutoff levels.
Parameters:
- base_path must point to the base directory with swref/ directory containing
the reference SW generated using GTTl and processed by gttl_parser.py
- file1 and file2 point to the target and query FASTA respectively.
- k_vals, thresholds_vals, tolerance_vals and min_frag_lens are the input
parameter ranges
The script outputs to stdout in tsv format: reference cutoff, k, threshold, tolerance, min_len, recall, precision, f1-score
"""

import subprocess
import numpy as np
import re

base_path = "../"
cutoff_vals = ['47.0', "48.0", "50.0", "53.0",
               "56.0", "60.0", "66.0", "74.0", "86.0", "110.0"]
reffile = base_path + "swref/cutoff_{}.txt"
file1 = base_path + "ma_data/uniref50_200_512_1m_0_1k.fasta"
file2 = base_path + "ma_data/uniref50_200_512_1m_1_100k.fasta"
query_exec = base_path + "ma_src/query.x"
pr_exec = base_path + "ma_scripts/segsearchflat_vs_sw.py"
# k_vals = [i for i in range(2, 10, 2)]
k_vals = [1]
threshold_vals = np.linspace(0.75, 0.9, 4)
tolerance_vals = [i for i in range(3, 4, 1)]
min_frag_lens = [i for i in range(2, 3, 1)]
out = []

print(k_vals, threshold_vals, tolerance_vals, min_frag_lens)


def get_f1_score(r, p):
    return 2 * r * p / (r + p)


for k in k_vals:
    for t in threshold_vals:
        for tolerance in tolerance_vals:
            for l in min_frag_lens:
                query_out = base_path + \
                    "measurements2/cache/out_{}_{:.3f}_{}_{}.tsv".format(
                        k, t, tolerance, l)
                query_args = [
                    query_exec, "-g",
                    "-m",  base_path + "ckpts.pt",
                    "-k", str(k),
                    "-t", str(t),
                    "-l", str(tolerance),
                    "-f", str(l),
                    "-i", base_path + "index",
                    "-s",
                    "--query_batch_size", str(200),
                    file2, file1
                ]
                print(' '.join(query_args))
                query_result = subprocess.run(
                    query_args,
                    capture_output=True,
                    text=True
                )

                with open(query_out, "w") as f:
                    f.write(query_result.stdout)

                for cutoff in cutoff_vals:
                    rfile = reffile.format(cutoff)
                    pr_result = subprocess.run([
                        "python", pr_exec,
                        query_out,
                        rfile],
                        capture_output=True,
                        text=True
                    )
                    pr_out = base_path + \
                        "measurements2/cache/pr_{}_{}_{:.3f}_{}_{}.tsv".format(
                            cutoff, k, t, tolerance, l)
                    with open(pr_out, "w") as f:
                        f.write(pr_result.stdout)

                    with open(pr_out, "r") as f:
                        lines = list(f)
                        recall = float(
                            re.search(r"-?\d+(\.\d+)?", lines[0]).group(1))
                        precision = float(
                            re.search(r"-?\d+(\.\d+)?", lines[1]).group(1))
                        out.append([cutoff, k, float(
                            t), tolerance, l, recall, precision, get_f1_score(recall, precision)])
print("\n".join(["\t".join([str(v) for v in o]) for o in out]))

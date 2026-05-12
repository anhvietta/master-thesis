"""
compare ProtSearch result with MMseqs2, only on true positives
"""

import sys
# from scipy import stats
# from sklearn.metrics import jaccard_score
# import numpy as np
# import tables as tb
from Bio import SeqIO

ss_file = sys.argv[1]
sw_file = sys.argv[2]
mmseqs2_file = sys.argv[3]
target_file = sys.argv[4]
query_file = sys.argv[5]

target_fasta = SeqIO.index(target_file, "fasta")
target_headers = list(target_fasta.keys())
query_fasta = SeqIO.index(query_file, "fasta")
query_headers = list(query_fasta.keys())

with open(ss_file, 'r') as handler:
    ss_data = [line[:-1].split("\t") for line in list(handler)[41:]]

with open(sw_file, 'r') as handler:
    sw_data = [line[:-1].split("\t") for line in list(handler)]

with open(mmseqs2_file, 'r') as handler:
    mmseqs2_data = [line[:-1].split("\t") for line in list(handler)]

ss_seqpair_set = set(
    (target_headers[int(e[0])], query_headers[int(e[1])]) for e in ss_data)
sw_seqpair_set = set((e[0], e[1]) for e in sw_data)
mmseqs2_seqpair_set = set((e[1], e[0]) for e in mmseqs2_data)

# ss_dict = {(e[0], e[1]) : [float(v) for v in e[2:-2]] + e[-2:] for e in ss_data}
# sw_dict = {(e[0], e[1]) : [float(v) for v in e[2:-2]] + e[-2:] for e in sw_data}

ss_intersection = ss_seqpair_set.intersection(sw_seqpair_set)
mmseqs2_intersection = mmseqs2_seqpair_set.intersection(sw_seqpair_set)
print(len(ss_intersection), len(mmseqs2_intersection))
intersection = ss_intersection.intersection(mmseqs2_intersection)
print("Recall: ", len(intersection) / len(mmseqs2_intersection))
print("Precision: ", len(intersection) / len(ss_intersection))

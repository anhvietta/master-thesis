"""
Generate diagonal measurements
"""

import sys
# from scipy import stats
# from sklearn.metrics import jaccard_score
# import numpy as np
import torch
# import tables as tb

ss_file = sys.argv[1]
sw_file = sys.argv[2]

with open(ss_file, 'r') as handler:
    ss_data = [line.strip().split("\t") for line in list(handler)[41:]]

with open(sw_file, 'r') as handler:
    sw_data = [line.strip().split("\t") for line in list(handler)]

ss_seqpair_set = set((e[0], e[1]) for e in ss_data)
sw_seqpair_set = set((e[0], e[1]) for e in sw_data)

ss_dict = {(e[0], e[1]): e[-2:] for e in ss_data}
sw_dict = {(e[0], e[1]): e[-2:] for e in sw_data}

intersection = ss_seqpair_set.intersection(sw_seqpair_set)
# print("Recall: ", len(intersection) / len(sw_seqpair_set))
# print("Precision: ", len(intersection) / len(ss_seqpair_set))

ss_filtered = {}
sw_filtered = {}

ss_dict_filtered = {k: ss_dict[k][-2:] for k in intersection}
sw_dict_filtered = {k: sw_dict[k][-2:] for k in intersection}

for k in intersection:
    if len(ss_dict_filtered[k][0]) != len(sw_dict_filtered[k][0]) or len(ss_dict_filtered[k][1]) != len(sw_dict_filtered[k][1]):
        print(k)
        print(ss_dict_filtered[k][0], sw_dict_filtered[k][0])
        print(ss_dict_filtered[k][1], sw_dict_filtered[k][1])
    assert (len(ss_dict_filtered[k][0]) == len(sw_dict_filtered[k][0]) and len(
        ss_dict_filtered[k][1]) == len(sw_dict_filtered[k][1]))


def make_map(v_t, v_q):
    vt = torch.tensor([c == "1" for c in v_t]).bool()
    vq = torch.tensor([c == "1" for c in v_q]).bool()
    # print(vt, vq)
    l = vt.shape[0]
    n = vq.shape[0]
    m = torch.zeros((l, n), dtype=torch.bool)
    tidxs = torch.argwhere(vt)
    qidxs = torch.argwhere(vq)
    for tidx, qidx in zip(tidxs, qidxs):
        m[tidx, qidx] = True
    # print(m.sum())
    return m


def map_pr(pred_mask, ref_mask, eps=1e-8):
    pred_mask = pred_mask.bool()
    ref_mask = ref_mask.bool()
    # print(pred_mask.sum(), ref_mask.sum())

    # True positives: predicted 1 and actually 1
    tp = (pred_mask & ref_mask).sum().float()

    # False positives: predicted 1 but actually 0
    fp = (pred_mask & ~ref_mask).sum().float()

    # False negatives: predicted 0 but actually 1
    fn = (~pred_mask & ref_mask).sum().float()
    # print(tp, fp, fn)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return precision, recall, tp, fp, fn


def get_diag(v_t, v_q):
    vt = torch.tensor([c == "1" for c in v_t]).bool()
    vq = torch.tensor([c == "1" for c in v_q]).bool()
    assert (vt.sum() == vq.sum())
    vt_pos = torch.argwhere(vt)
    vq_pos = torch.argwhere(vq)
    zipped_pos = [(v[0].item(), v[1].item()) for v in zip(vt_pos, vq_pos)]
    start = (None, None)
    end = (None, None)
    diags = []
    for m in zipped_pos:
        if not start[0]:
            start = m
            continue
        end = m
        if m[0] - start[0] != m[1] - start[1]:
            diags.append((start[0] - start[1], start[0], end[0]))
            start = m
    if not end[0]:
        diags.append((start[0] - start[1], start[0], start[0]))
    else:
        diags.append((start[0] - start[1], start[0], end[0]))
    return diags


diagsref_data = []
diags_data = []


def diag_include(ref, d, overlap_f=0.5):
    if ref[1] <= d[0] or d[1] <= ref[0]:
        return False
    overlap = (max(ref[0], d[0]), min(ref[1], d[1]))
    # print(overlap, ref, d, overlap[1] - overlap[0] > overlap_f * (ref[1] - ref[0]))
    return overlap[1] - overlap[0] > overlap_f * (ref[1] - ref[0])


precision_total, recall_total = 0, 0
tp_total, fp_total, fn_total = 0, 0, 0
for pair_id in list(intersection):
    # print(sw_dict_filtered[pair_id], ss_dict_filtered[pair_id])
    refmap = make_map(*sw_dict_filtered[pair_id])
    m = make_map(*ss_dict_filtered[pair_id])
    # print(m, refmap)
    diags_ref = get_diag(*sw_dict_filtered[pair_id])
    diags = get_diag(*ss_dict_filtered[pair_id])
    for diagref in diags_ref:
        diagsref_data.append(diagref[2] - diagref[1] + 1)
        for diag in diags:
            # if diagref[2] - diagref[1] + 1 > 370:
            #    print(diagref, sw_dict_filtered[pair_id], diag, ss_dict_filtered[pair_id])
            if diagref[0] - diag[0] == 0 and diag_include(diagref[1:], diag[1:]):
                diags_data.append(diagref[2] - diagref[1] + 1)
    precision, recall, tp, fp, fn = map_pr(m, refmap)
    precision_total += precision.item()
    recall_total += recall.item()
    tp_total += tp
    fp_total += fp
    fn_total += fn
    print(f"{pair_id[0]}\t{pair_id[1]}\t{precision.item()}\t{recall.item()}")

precision = tp_total / (tp_total + fp_total)
recall = tp_total / (tp_total + fn_total)
print(precision.item(), recall.item())

precision_total = precision_total / len(intersection)
recall_total = recall_total / len(intersection)
print(precision_total, recall_total)

print(diagsref_data, diags_data)

# res = stats.pearsonr([e[5-2] for e in ss_dict_filtered.values()], [e[6-2] for e in sw_dict_filtered.values()])
# print(res.statistic, res.pvalue)

'''s = 0
for k,v in sw_dict_filtered.items():
    s += jaccard_score([int(c) for c in v[-1]], [int(c) for c in ss_dict_filtered[k][-1]])
    s += jaccard_score([int(c) for c in v[-2]], [int(c) for c in ss_dict_filtered[k][-2]])
print(s / len(sw_dict_filtered) / 2)

diff_set = sw_seqpair_set.difference(ss_seqpair_set)
dtype = np.dtype([
    ('u1', 'u4'),
    ('u2', 'u4'),
    ('u3', 'u4'),
    ('u4', 'u4'),
    ('u5', 'u4'),
    ('u6', 'u4'),
    ('u7', 'u8'),
    ('str1_bits', np.uint8, (64,)),
    ('str2_bits', np.uint8, (64,)),
])
diff_set_size = len(diff_set)
mat = np.recarray((diff_set_size,),dtype=dtype)

for i, (target_seqnum, query_seqnum) in enumerate(diff_set):
    entry = sw_dict[(target_seqnum, query_seqnum)]
    ts, te, qs, qe =  int(entry[0]), int(entry[0]) + int(entry[1]), int(entry[2]), int(entry[2]) + int(entry[3])
    mat[i]['u1'] = int(target_seqnum)
    mat[i]['u2'] = int(query_seqnum) + 10000
    mat[i]['u3'] = ts + 1
    mat[i]['u4'] = te
    mat[i]['u5'] = qs + 1
    mat[i]['u6'] = qe
    mat[i]['u7'] = int(entry[4])
    mat[i]['str1_bits'] = np.packbits(np.array([c == "1" for c in entry[-2]] + [0] * (512 - len(entry[-2])),dtype=bool))
    mat[i]['str2_bits'] = np.packbits(np.array([c == "1" for c in entry[-1]] + [0] * (512 - len(entry[-1])),dtype=bool))

def write(outfile, data):
    aln_h5f = tb.open_file(outfile,"a")
    table = aln_h5f.create_table('/', 'aln_data', description=dtype, title="ALN Table")
    #for i, (q_idx, qaln_data) in enumerate(intdata.items()):
    table.append(data)
    aln_h5f.close()

outfile = sys.argv[3]
write(outfile, mat)
'''
# ./sw_all_against_all.x -d ../../../masterarbeit-anhvietta/data/tmp_10000_target.fasta -q ../../../masterarbeit-anhvietta/data/tmp_100_query.fasta -v 2 -c 47 -t 6 -a 4+512 -o ../../../masterarbeit-anhvietta/swref/100_10000/out

#!/usr/bin/env python3
"""
Convert a MMseqs2 output file to a binary tabular format
Note: positive file not needed
"""

import sys
import argparse
import time
import random
import h5py
import numpy as np
import pandas as pd
import tables as tb
from Bio import SeqIO


def parse_command_line(argv):
    p = argparse.ArgumentParser(description='Split a hdf5 dataset')
    p.add_argument('-d', '--debug', action='store_true', default=False,
                   help='show debug output')
    p.add_argument("-p", "--positive", nargs=1, type=str,
                   required=True,  help="Specify the input file for positive samples.")
    p.add_argument("-n", "--negative", nargs=1, type=str,
                   required=True,  help="Specify the input file for negative samples.")
    p.add_argument("-q", "--query", nargs=1, type=str,
                   required=True,  help="Specify the query file.")
    p.add_argument("-t", "--target", nargs=1, type=str,
                   required=True,  help="Specify the target file.")
    p.add_argument('-o', '--output', nargs='+', default='./',
                   help='path to output directory')
    return p.parse_args(argv)


def make_local_aln_mat(qaln, taln, qstart, qend, tstart, tend, max_len=512):
    def score_aa(a, b):
        if a == '-':
            assert (b != '-')
            return True, (0, 1)
        elif b == '-':
            return True, (1, 0)
        elif a != b:
            return True, (1, 1)
        else:
            return True, (1, 1)
    qstart, qend, tstart, tend = int(qstart), int(qend), int(tstart), int(tend)
    assert (len(qaln) == len(taln))
    assert (all(coord - 1 < max_len and coord >
            0 for coord in [qstart, qend, tstart, tend]))

    aln = np.zeros((tend - tstart + 1, qend - qstart + 1),
                   dtype=np.dtype(bool))
    coord_q = 0
    coord_t = 0
    for i, c in enumerate(qaln):
        score, (mov_q, mov_t) = score_aa(c, taln[i])
        aln[coord_t, coord_q] = score
        coord_t += mov_t
        coord_q += mov_q
    assert (coord_q == qend - qstart + 1 and coord_t == tend - tstart + 1)
    return aln


'''def make_aln_bitvector(qaln, taln):
    assert(len(qaln) == len(taln))
    vq = [a != '-' for a in qaln]
    vt = [a != '-' for a in taln]
    vq = np.packbits(vq)
    vt = np.packbits(vt)
    return (vq,vt)'''


def make_aln_bitvector(qaln, taln, qs, qe, ts, te, max_len=512):
    qs, qe, ts, te = int(qs), int(qe), int(ts), int(te)
    assert (len(qaln) == len(taln))
    assert (all(coord - 1 < max_len and coord >
            0 for coord in [qs, qe, ts, te]))
    vq = np.zeros((512), dtype=bool)
    vt = np.zeros((512), dtype=bool)
    qi = qs - 1
    ti = ts - 1
    for a, b in zip(qaln, taln):
        if a == '-':
            assert (b != '-')
            ti += 1
        elif b == '-':
            qi += 1
        else:
            vq[qi] = True
            vt[ti] = True
            qi += 1
            ti += 1
    assert (qi == qe and ti == te)
    vq = np.packbits(vq)
    vt = np.packbits(vt)
    return vq, vt


def parse_matches(inputfile):
    matches = {}
    with open(inputfile, 'r') as file:
        for line in file:
            split = line[:-1].split('\t')
            assert (len(split) == 21)
            q_id = split[0]
            t_id = split[1]

            if q_id not in matches.keys():
                matches[q_id] = {}
            vq, vt = make_aln_bitvector(*split[5:11])
            matches[q_id][t_id] = split[7:11] + [split[-2]] + [vq, vt]

    return matches


def parse_positive(inputfile):
    positive = {}
    with open(inputfile, 'r') as file:
        for line in file:
            split = line[:-1].split('\t')
            assert (len(split) == 21)
            q_id = split[0]
            t_id = split[1]

            if q_id not in positive.keys():
                positive[q_id] = {}
            vq, vt = make_aln_bitvector(*split[5:11])
            positive[q_id][t_id] = split[7:11] + [vq, vt]

    return positive


def sample_negatives(inputfile, positives):
    match_dict = {}
    negatives = {}
    with open(inputfile, 'r') as file:
        for line in file:
            split = line[:-1].split('\t')
            assert (len(split) == 21)
            q_id = split[0]
            t_id = split[1]

            if q_id not in positives.keys():
                continue
            if t_id in positives[q_id]:
                continue

            if q_id not in negatives.keys():
                negatives[q_id] = {}
            vq, vt = make_aln_bitvector(*split[5:11])
            negatives[q_id][t_id] = split[7:11] + [vq, vt]
    for q_id in list(negatives.keys()):
        assert (q_id in positives.keys())
        if len(negatives[q_id].keys()) < len(positives[q_id].keys()):
            del positives[q_id]
            del negatives[q_id]
    for q_id in set(positives.keys()).difference(set(negatives.keys())):
        del positives[q_id]
    return negatives


def make_matches(matches, query_seqs, target_seqs,  outfile, limit=100000):
    cnt = 0
    seqids = {}
    match_data = {}
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

    def write(outfile, cnt, intdata):
        aln_h5f = tb.open_file(outfile, "a")
        if '/aln_data' in aln_h5f:
            table = aln_h5f.get_node('/aln_data')
        else:
            table = aln_h5f.create_table(
                '/', 'aln_data', description=dtype, title="ALN Table")

        for i, (q_idx, qaln_data) in enumerate(intdata.items()):
            table.append(qaln_data)
        aln_h5f.close()

        print(f"Append {cnt} matches to {outfile}")

    def process_data(data, q_idx):
        total_len = len(data)
        mat = np.recarray((total_len,), dtype=dtype)
        mat['u1'] = q_idx
        for i, (entry, entry_data) in enumerate(data.items()):
            assert (entry in target_seqs)
            if entry not in seqids:
                seqids[entry] = (len(seqids), False)
            idx, _ = seqids[entry]
            mat[i]['u2'] = idx
            mat[i]['u3'] = entry_data[0]
            mat[i]['u4'] = entry_data[1]
            mat[i]['u5'] = entry_data[2]
            mat[i]['u6'] = entry_data[3]
            mat[i]['u7'] = entry_data[4]
            mat[i]['str1_bits'] = entry_data[5]
            mat[i]['str2_bits'] = entry_data[6]
        return mat

    for q_id, q_data in matches.items():
        assert (q_id in query_seqs)
        if q_id not in seqids:
            seqids[q_id] = (len(seqids), True)
        q_idx, _ = seqids[q_id]

        match_data[q_idx] = process_data(q_data, q_idx)

        cnt += 1
        if cnt >= limit:
            write(outfile+'_matches.h5', cnt, match_data)

            cnt = 0
            match_data = {}

    if cnt > 0:
        write(outfile+'_matches.h5', cnt, match_data)
    return seqids


def make_match(positives, negatives, query_seqs, target_seqs,  outfile, limit=100000):
    cnt = 0
    seqids = {}
    assert (set(positives.keys()) == set(negatives.keys()))
    negative_data = {}
    positive_data = {}
    dtype = np.dtype([
        ('u1', 'u4'),
        ('u2', 'u4'),
        ('u3', 'u4'),
        ('u4', 'u4'),
        ('u5', 'u4'),
        ('u6', 'u4'),
        ('str1_bits', np.uint8, (64,)),
        ('str2_bits', np.uint8, (64,)),
    ])

    def write(outfile, cnt, intdata):
        aln_h5f = tb.open_file(outfile, "a")
        if '/aln_data' in aln_h5f:
            table = aln_h5f.get_node('/aln_data')
        else:
            table = aln_h5f.create_table(
                '/', 'aln_data', description=dtype, title="ALN Table")

        for i, (q_idx, qaln_data) in enumerate(intdata.items()):
            table.append(qaln_data)
        aln_h5f.close()

        print(f"Append {cnt} matches to {outfile}")

    def process_data(data, q_idx):
        total_len = len(data)
        mat = np.recarray((total_len,), dtype=dtype)
        mat['u1'] = q_idx
        for i, (entry, entry_data) in enumerate(data.items()):
            assert (entry in target_seqs)
            if entry not in seqids:
                seqids[entry] = (len(seqids), False)
            idx, _ = seqids[entry]
            mat[i]['u2'] = idx
            mat[i]['u3'] = entry_data[0]
            mat[i]['u4'] = entry_data[1]
            mat[i]['u5'] = entry_data[2]
            mat[i]['u6'] = entry_data[3]
            mat[i]['str1_bits'] = entry_data[4]
            mat[i]['str2_bits'] = entry_data[5]
        return mat

    for q_id, q_data in negatives.items():
        assert (q_id in query_seqs)
        if q_id not in seqids:
            seqids[q_id] = (len(seqids), True)
        q_idx, _ = seqids[q_id]

        negative_data[q_idx] = process_data(q_data, q_idx)
        positive_data[q_idx] = process_data(positives[q_id], q_idx)

        cnt += 1
        if cnt >= limit:
            write(outfile+'_negative.h5', cnt, negative_data)
            write(outfile+'_positive.h5', cnt, positive_data)

            cnt = 0
            negative_data = {}
            positive_data = {}
            neg_qaln = []
            neg_taln = []
            pos_qaln = []
            pos_taln = []

    if cnt > 0:
        write(outfile+'_negative.h5', cnt, negative_data)
        write(outfile+'_positive.h5', cnt, positive_data)
    return seqids


def make_fasta(seqids, query_index, target_index, outfile, limit=1000000):
    seqids_l = list(seqids.items())
    seqids_l.sort(key=lambda x: x[1][0])
    seqidx = [s[1][0] for s in seqids_l]
    assert (len(seqidx) == len(set(seqidx)))
    s = ''
    cnt = 0
    for seq_id, (i, is_query) in seqids_l:
        if is_query:
            seq = str(query_index[seq_id].seq)
        else:
            seq = str(target_index[seq_id].seq)
        assert (i % limit == cnt)
        s += f'>{seq_id}\n{seq}\n'
        cnt += 1
        if (i+1) % limit == 0:
            with open(outfile, 'a') as handler:
                handler.write(s)
            print(f"Wrote {cnt} sequences to {outfile}")
            cnt = 0
            s = ''
    if s != '':
        assert (cnt > 0)
        with open(outfile, 'a') as handler:
            handler.write(s)
        print(f"Wrote {cnt} sequences to {outfile}")


if __name__ == '__main__':
    args = parse_command_line(sys.argv[1:])
    t = time.time()
    query_index = SeqIO.index(args.query[0], "fasta")
    target_index = SeqIO.index(args.target[0], "fasta")
    query_seqs = set(list(query_index.keys()))
    target_seqs = set(list(target_index.keys()))
    print(f'Time to index: {time.time()-t}')

    '''positives = parse_positive(args.positive[0])
    print(f'Time to parse positive: {time.time()-t}')
    negatives = sample_negatives(args.negative[0], positives)
    print(f'Time to sample negatives: {time.time()-t}')

    seqids = make_match(positives, negatives, query_seqs, target_seqs, args.output[0])

    make_fasta(seqids,query_index,target_index, args.output[0]+'.fasta')
    print(f'Time to create db: {time.time()-t}')'''

    matches = parse_matches(args.negative[0])
    seqids = make_matches(matches, query_seqs, target_seqs, args.output[0])
    make_fasta(seqids, query_index, target_index, args.output[0]+'.fasta')

"""
This module contains the dataset class for the fasta dataset.
"""

import torch
import numpy as np
import h5py
import random
import json
from pathlib import Path
from fasta_reader import read_fasta
from torch.utils.data import Dataset
from utils import create_encoding_one_hot, create_encoding, create_encoding_with_len
from Bio import SeqIO
from constants import max_length, BYOL_Const
import tables
import jax.numpy as jnp

class SequenceDataset(Dataset):
    def __init__(self, fasta_file, h5file):
        if h5file:
            with h5py.File(h5file, 'r') as file:
                self.sequences = file.get('encoding')[()]
                self.seqlen = file.get('seqlen')[()]
        else:
            fasta_dict = SeqIO.index(fasta_file,'fasta')
            self.sequences, self.seqlen = create_encoding_with_len(fasta_dict , max_length)
            '''self.export(h5file)
            print(f'Created cache at {h5file}')'''

    def __len__(self):
        return self.seqlen.shape[0]

    def __getitem__(self, idx):
        seq = self.sequences[idx, :]
        seqlen = self.seqlen[idx]
        return seq, seqlen

    def export(self, outfile):
        with h5py.File(outfile, 'w') as file:
            file.create_dataset('encoding', data=self.sequences)
            file.create_dataset('seqlen', data=self.seqlen)

class FastaDataset(Dataset):
    def __init__(self, fasta_file, h5file=None, max_length=max_length):
        self.fasta_dict = dict(read_fasta(fasta_file))
        if h5file and Path(h5file).is_file():
            with h5py.File(h5file, 'r') as file:
                self.sequences = file.get('encoding')[()]
        else:
            #self.fasta_dict = SeqIO.index(fasta_file,'fasta')
            self.sequences = create_encoding(self.fasta_dict, max_length)
            outname = h5file
            self.export(outname)
            print(f'Created cache at {outname}')

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32)

    def get_dict(self):
        return self.fasta_dict

    def export(self, outfile):
        with h5py.File(outfile, 'w') as file:
            file.create_dataset('encoding', data=self.sequences)

class ContrastDataset(Dataset):
    def __init__(self, fasta_file, h5file=None, max_length=max_length):
        self.fasta_dict = dict(read_fasta(fasta_file))
        if h5file and Path(h5file).is_file():
            with h5py.File(h5file, 'r') as file:
                self.sequences = file.get('encoding')[()]
                self.seq_len = file.get('seq_len')[()]
        else:
            #self.fasta_dict = SeqIO.index(fasta_file,'fasta')
            self.sequences, self.seq_len = create_encoding_with_len(self.fasta_dict, max_length)
            self.export(h5file)
            print(f'Created cache at {outname}')

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        return seq, self.augment(seq, self.seq_len[idx])

    def get_dict(self):
        return self.fasta_dict

    def export(self, outfile):
        with h5py.File(outfile, 'w') as file:
            file.create_dataset('encoding', data=self.sequences)
            file.create_dataset('seq_len', data=self.seq_len)

    def augment(self,seq,seq_len):
        # Remove pad tokens if present
        augmented = seq[:seq_len].detach().clone()

        frag_len = int(BYOL_Const.fragment_length_dist[torch.randint(len(fragment_length_dist), (1,))])
        if frag_len < seq_len:
            start_idx = random.randint(0, valid_len - frag_len)
            augmented = augmented[start_idx: start_idx + frag_len]

        del_len = int(BYOL_Const.deletion_length_dist[torch.randint(len(deletion_length_dist), (1,))])
        if del_len > 0 and del_len < len(seq):
            del_start = random.randint(0, len(seq) - del_len)
            augmented[del_start : delstart + del_len] = 0

        del_len = int(BYOL_Const.deletion_length_dist[torch.randint(len(deletion_length_dist), (1,))])
        if del_len > 0 and del_len < len(seq):
            del_start = random.randint(0, len(seq) - del_len)
            augmented[del_start : delstart + del_len] = 0

        return augmented

class ColBertDataset(Dataset):
    def __init__(self, fasta_file, positive_data, negative_data, h5file=None, max_length=max_length):
        if h5file and Path(h5file).is_file():
            with h5py.File(h5file, 'r') as file:
                self.sequences = file.get('encoding')[()]
        else:
            fasta_dict = SeqIO.index(fasta_file,'fasta')
            self.sequences = create_encoding(fasta_dict , max_length)
            self.export(h5file)
            print(f'Created cache at {h5file}')
        
        self.pos_matches, self.neg_matches = self.read_match_data(positive_data, negative_data)
        assert(all(i == j for i,j in zip(self.pos_matches.keys(),self.neg_matches.keys())))
        
        self.len = self.get_len()
        print(len(self.pos_matches), self.len)
        self.rng = np.random.default_rng()
        self.create_dataset()

    def get_len(self):
        pos_match_cnt = 0
        for matches in self.pos_matches.values():
            pos_match_cnt += matches.shape[0]
        return pos_match_cnt
    
    @staticmethod
    def read_match_data(positive_data, negative_data):
        def parse(file):
            d = {}
            with h5py.File(file,'r') as handler:
                for q_idx in handler.keys():
                    d[q_idx] = handler.get(q_idx)[()]
            return d
        pos_matches = parse(positive_data)
        neg_matches = parse(negative_data)
        return pos_matches, neg_matches

    def __len__(self):
        return self.len

    def span_is_valid(self,span):
        return span[0] < span[1]

    def coord_to_gmap(self, shape, center, sigma=0.1):
        x = np.arange(0, shape[1], 1, float)
        y = np.arange(0, shape[0], 1, float)[:, np.newaxis]
        heatmap = np.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * sigma**2))
        return heatmap

    def __getitem__(self, idx):
        data = self.dataset[idx,:]
        q_idx = data[0]
        p_idx = data[1]
        for span in [data[2:4],data[4:6],data[7:9],data[9:11]]:
            assert(self.span_is_valid(span))
        for i in [2,3,4,5,7,8,9,10]:
            assert(data[i] > 0)
        qpspan = (data[2]-1,data[3]-1)
        pspan = (data[4]-1,data[5]-1)
        n_idx = data[6]
        qnspan = (data[7]-1,data[8]-1)
        nspan = (data[9]-1,data[10]-1)

        gmap_shape = (max_length, max_length)
        p_start_coord = (data[2]-1,data[4]-1)
        p_end_coord = (data[3]-1,data[5]-1)
        n_start_coord = (data[7]-1,data[9]-1)
        n_end_coord = (data[8]-1,data[10]-1)

        pstart_gmap = self.coord_to_gmap(gmap_shape, p_start_coord)
        pend_gmap = self.coord_to_gmap(gmap_shape, p_end_coord)
        pgmap = np.stack([pstart_gmap, pend_gmap], axis=0).astype(np.float32)
        nstart_gmap = self.coord_to_gmap(gmap_shape, n_start_coord)
        nend_gmap = self.coord_to_gmap(gmap_shape, n_end_coord)
        ngmap = np.stack([nstart_gmap, nend_gmap], axis=0).astype(np.float32)

        query_seq = self.sequences[q_idx, :]
        pos_seq = self.sequences[p_idx, :]
        neg_seq = self.sequences[n_idx, :]
        return query_seq, pos_seq, qpspan, pspan, neg_seq, qnspan, nspan, torch.tensor(pgmap), torch.tensor(ngmap)

    def export(self, outfile):
        with h5py.File(outfile, 'w') as file:
            file.create_dataset('encoding', data=self.sequences)

    def sample_negative(self):
        i = 0
        for q_idx, match in self.neg_matches.items():
            self.rng.shuffle(match)
            pos_len = self.pos_matches[q_idx].shape[0]
            assert(match.shape[0] >= pos_len)
            self.dataset[i:i+pos_len, 6:11] = match[0:pos_len,:]
            i += pos_len

    def create_dataset(self):
        n = self.len
        self.dataset = np.empty((n,11), dtype=np.uint32)
        i = 0
        for q_idx, match in self.pos_matches.items():
            pos_len = match.shape[0]
            self.dataset[i:i+pos_len, 0] = q_idx
            self.dataset[i:i+pos_len, 1:6] = match[:,:]
            i += pos_len
        self.sample_negative()

class ColBertDatasetAln(Dataset):
    def __init__(self, fasta_file, positive_data, negative_data, h5file=None, max_length=max_length, padding=0):
        self.padding=padding
        if h5file and Path(h5file).is_file():
            with h5py.File(h5file, 'r') as file:
                self.sequences = file.get('encoding')[()]
                self.seqlen = file.get('seqlen')[()]
        else:
            fasta_dict = SeqIO.index(fasta_file,'fasta')
            self.sequences, self.seqlen = create_encoding_with_len(fasta_dict , max_length)
            self.export(h5file)
            print(f'Created cache at {h5file}')

        self.dtype = np.dtype([
            ('u1', 'u4'),
            ('u2', 'u4'),
            ('u3', 'u4'),
            ('u4', 'u4'),
            ('u5', 'u4'),
            ('u6', 'u4'),
            ('str1_bits', np.uint8, (64,)),
            ('str2_bits', np.uint8, (64,)),
            ('u7', 'u4'),
            ('u8', 'u4'),
            ('u9', 'u4'),
            ('u10', 'u4'),
            ('u11', 'u4'),
            ('str3_bits', np.uint8, (64,)),
            ('str4_bits', np.uint8, (64,))
        ])

        self.pos_key = ['u1','u2','u3','u4','u5','u6','str1_bits','str2_bits']
        self.neg_key = {
            'u7':'u2',
            'u8':'u3',
            'u9':'u4',
            'u10':'u5',
            'u11':'u6',
            'str3_bits':'str1_bits',
            'str4_bits':'str2_bits'
        }

        self.pos_matches, self.neg_matches, self.neg_info = self.read_match_data(positive_data, negative_data)
        self.rng = np.random.default_rng()
        self.create_dataset()

    @staticmethod
    def read_match_data(positive_data, negative_data):
        def parse(f):
            h5file = tables.open_file(f, mode="r")
            table = h5file.root.aln_data
            data = table.read()
            h5file.close()
            return data

        pos_table = parse(positive_data)
        neg_table = parse(negative_data)
        curr_idx = -1
        neg_info = {}
        for i, row in enumerate(neg_table):
            idx = row['u1']
            b = idx in neg_info
            if curr_idx != idx and b:
                # new id but already added => wrong order in file
                raise RuntimeError("Mixed order")
            elif curr_idx != idx:
                # new id, add to info, correct end
                assert(curr_idx == -1 or curr_idx in neg_info)
                if curr_idx != -1:
                    neg_info[curr_idx]['end'] = i
                neg_info[idx] = {
                    'start': i,
                    'end': None,
                    'pos_length': None
                }
                curr_idx = idx
        neg_info[curr_idx]['end'] = len(neg_table)
        curr_idx = -1
        start = 0
        for i, row in enumerate(pos_table):
            idx = row['u1']
            assert(idx in neg_info)
            if idx != curr_idx:
                if curr_idx != -1:
                    neg_info[curr_idx]['pos_length'] = i - start
                curr_idx = idx
                start = i
        neg_info[curr_idx]['pos_length'] = len(pos_table) - start
        for d in neg_info.values():
            assert(d['pos_length'] is not None)
            assert(d['end'] is not None)
        return pos_table, neg_table, neg_info

    def __len__(self):
        return self.pos_matches.shape[0]

    def span_is_valid(self,span):
        return span[0] < span[1] and span[0] >= 1 and span[1] <= span[2]

    def coord_to_gmap(self, shape, center, sigma=2):
        x = np.arange(0, shape[1], 1, float)
        y = np.arange(0, shape[0], 1, float)[:, np.newaxis]
        heatmap = np.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * sigma**2))
        return heatmap

    def __getitem__(self, idx):
        data = self.dataset[idx]
        q_idx = data['u1']
        p_idx = data['u2']
        n_idx = data['u7']

        query_seq = self.sequences[q_idx, :]
        pos_seq = self.sequences[p_idx, :]
        neg_seq = self.sequences[n_idx, :]
        query_len = self.seqlen[q_idx]
        pos_len = self.seqlen[p_idx]
        neg_len = self.seqlen[n_idx]

        for span in [[data['u3'],data['u4'],query_len],[data['u5'],data['u6'], pos_len],[data['u8'],data['u9'], query_len],[data['u10'],data['u11'],neg_len]]:
            assert(self.span_is_valid(span))
        qpspan = [data['u3']-1,data['u4']-1]
        pspan = [data['u5']-1,data['u6']-1]
        qnspan = [data['u8']-1,data['u9']-1]
        nspan = [data['u10']-1,data['u11']-1]
        span = np.stack((qpspan, pspan, qnspan, nspan)).astype(np.int64)

        gmap_shape = (max_length, max_length)
        p_start_coord = (data['u3']-1,data['u5']-1)
        p_end_coord = (data['u4']-1,data['u6']-1)
        n_start_coord = (data['u8']-1,data['u10']-1)
        n_end_coord = (data['u9']-1,data['u11']-1)

        pstart_gmap = self.coord_to_gmap(gmap_shape, p_start_coord)
        pend_gmap = self.coord_to_gmap(gmap_shape, p_end_coord)
        nstart_gmap = self.coord_to_gmap(gmap_shape, n_start_coord)
        nend_gmap = self.coord_to_gmap(gmap_shape, n_end_coord)
        maps = np.stack([pstart_gmap, pend_gmap, nstart_gmap, nend_gmap], axis=0).astype(np.float32)

        qpaln = torch.tensor(np.unpackbits(data['str1_bits']), dtype=torch.bool)
        paln = torch.tensor(np.unpackbits(data['str2_bits']), dtype=torch.bool)
        qnaln = torch.tensor(np.unpackbits(data['str3_bits']), dtype=torch.bool)
        naln = torch.tensor(np.unpackbits(data['str4_bits']), dtype=torch.bool)
        aln = np.stack([qpaln, paln, qnaln, naln], axis=0).astype(bool)

        return np.stack((query_seq, pos_seq, neg_seq), axis=0).astype(np.uint8), np.array([query_len, pos_len, neg_len], dtype=np.float32), span, maps, aln, idx

    def export(self, outfile):
        with h5py.File(outfile, 'w') as file:
            file.create_dataset('encoding', data=self.sequences)
            file.create_dataset('seqlen', data=self.seqlen)

    def sample_negative(self):
        curr_idx = -1
        i = 0
        for row in self.dataset:
            idx = row['u1']
            if idx != curr_idx:
                curr_idx = idx
                start = self.neg_info[idx]['start']
                end = self.neg_info[idx]['end']
                length = self.neg_info[idx]['pos_length']
                match = self.neg_matches[start:end]
                self.rng.shuffle(match)
                for k1, k2 in self.neg_key.items():
                    self.dataset[i:i+length][k1] = match[:length][k2]
                i += length
        assert(i == len(self.dataset))

    def create_dataset(self):
        n = self.pos_matches.shape[0]
        self.dataset = np.empty((n,), dtype=self.dtype)
        for k in self.pos_key:
            self.dataset[k][:] = self.pos_matches[k][:]
        self.sample_negative()

class ColBertDatasetAln_direct(Dataset):
    def __init__(self, fasta_file, match_data, h5file=None, max_length=max_length, padding=0, index1=False):
        self.padding=padding
        if h5file and Path(h5file).is_file():
            with h5py.File(h5file, 'r') as file:
                self.sequences = file.get('encoding')[()]
                self.seqlen = file.get('seqlen')[()]
        else:
            fasta_dict = SeqIO.index(fasta_file,'fasta')
            self.sequences, self.seqlen = create_encoding_with_len(fasta_dict , max_length)
            self.export(h5file)
            print(f'Created cache at {h5file}')

        self.offset = 0 if not index1 else 1

        self.dtype = np.dtype([
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

        self.key = ['u1','u2','u3','u4','u5','u6','u7','str1_bits','str2_bits']

        matches = self.read_match_data(match_data)
        idx = np.argsort(matches['u7'])
        self.dataset = matches[idx]

    @staticmethod
    def read_match_data(match_data):
        def parse(f):
            h5file = tables.open_file(f, mode="r")
            table = h5file.root.aln_data
            data = table.read()
            h5file.close()
            return data
        table = parse(match_data)
        return table

    def __len__(self):
        return self.dataset.shape[0]

    def span_is_valid(self,span,lo,hi):
        return span[0] < span[1] and span[0] >= lo and span[1] <= hi

    def __getitem__(self, idx):
        data = self.dataset[idx]
        q_idx = data['u1']
        t_idx = data['u2']

        query_seq = self.sequences[q_idx, :]
        target_seq = self.sequences[t_idx, :]
        query_len = self.seqlen[q_idx]
        target_len = self.seqlen[t_idx]

        for span in [[data['u3'],data['u4'],query_len],[data['u5'],data['u6'], target_len]]:
            if not self.span_is_valid(span,lo=self.offset,hi=span[2]-1+self.offset):
                print(span)
            assert(self.span_is_valid(span,lo=self.offset,hi=span[2]-1+self.offset))

        qspan = [data['u3']-self.offset,data['u4']-self.offset]
        tspan = [data['u5']-self.offset,data['u6']-self.offset]
        span = np.stack((qspan, tspan)).astype(np.int64)

        qaln = torch.tensor(np.unpackbits(data['str1_bits']), dtype=torch.bool)
        taln = torch.tensor(np.unpackbits(data['str2_bits']), dtype=torch.bool)
        aln = np.stack([qaln, taln], axis=0).astype(bool)

        score = np.array([data['u7']], dtype=np.float32)

        return np.stack((query_seq, target_seq), axis=0).astype(np.uint8), np.array([query_len, target_len], dtype=np.float32), span, aln, score, idx

    def export(self, outfile):
        with h5py.File(outfile, 'w') as file:
            file.create_dataset('encoding', data=self.sequences)
            file.create_dataset('seqlen', data=self.seqlen)


class ValDataset(Dataset):
    def __init__(self, fasta_file, match_data, h5file=None, max_length=max_length, padding=0, index1=False):
        self.padding=padding
        fasta_dict = SeqIO.index(fasta_file,'fasta')
        self.headers = list(fasta_dict.keys())
        if h5file and Path(h5file).is_file():
            with h5py.File(h5file, 'r') as file:
                self.sequences = file.get('encoding')[()]
                self.seqlen = file.get('seqlen')[()]
        else:
            self.sequences, self.seqlen = create_encoding_with_len(fasta_dict , max_length)
            self.export(h5file)
            print(f'Created cache at {h5file}')

        self.offset = 0 if not index1 else 1

        self.dtype = np.dtype([
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

        self.key = ['u1','u2','u3','u4','u5','u6','u7','str1_bits','str2_bits']

        matches = self.read_match_data(match_data)
        idx = np.argsort(matches['u7'])
        self.dataset = matches[idx]

    @staticmethod
    def read_match_data(match_data):
        def parse(f):
            h5file = tables.open_file(f, mode="r")
            table = h5file.root.aln_data
            data = table.read()
            h5file.close()
            return data
        table = parse(match_data)
        return table

    def __len__(self):
        return self.dataset.shape[0]

    def span_is_valid(self,span,lo,hi):
        return span[0] < span[1] and span[0] >= lo and span[1] <= hi

    def __getitem__(self, idx):
        data = self.dataset[idx]
        q_idx = data['u1']
        t_idx = data['u2']

        query_seq = self.sequences[q_idx, :]
        target_seq = self.sequences[t_idx, :]
        query_len = self.seqlen[q_idx]
        target_len = self.seqlen[t_idx]

        for span in [[data['u3'],data['u4'],query_len],[data['u5'],data['u6'], target_len]]:
            if not self.span_is_valid(span,lo=self.offset,hi=span[2]-1+self.offset):
                print(span)
            assert(self.span_is_valid(span,lo=self.offset,hi=span[2]-1+self.offset))

        qspan = [data['u3']-self.offset,data['u4']-self.offset]
        tspan = [data['u5']-self.offset,data['u6']-self.offset]
        span = np.stack((qspan, tspan)).astype(np.int64)

        qaln = torch.tensor(np.unpackbits(data['str1_bits']), dtype=torch.bool)
        taln = torch.tensor(np.unpackbits(data['str2_bits']), dtype=torch.bool)
        aln = np.stack([qaln, taln], axis=0).astype(bool)

        score = np.array([data['u7']], dtype=np.float32)

        qheader = self.headers[q_idx]
        theader = self.headers[t_idx]

        return np.stack((query_seq, target_seq), axis=0).astype(np.uint8), np.array([query_len, target_len], dtype=np.float32), span, aln, score, idx, (qheader, theader)

    def export(self, outfile):
        with h5py.File(outfile, 'w') as file:
            file.create_dataset('encoding', data=self.sequences)
            file.create_dataset('seqlen', data=self.seqlen)

class JaxDataset:
    def __init__(self, fasta_file, match_data, h5file=None, max_length=max_length, index1=False):
        if h5file and Path(h5file).is_file():
            with h5py.File(h5file, 'r') as file:
                self.sequences = file.get('encoding')[()]
                self.seqlen = file.get('seqlen')[()]
        else:
            fasta_dict = SeqIO.index(fasta_file,'fasta')
            self.sequences, self.seqlen = create_encoding_with_len(fasta_dict , max_length)
            self.export(h5file)
            print(f'Created cache at {h5file}')
        self.offset = 0 if not index1 else 1

        self.dtype = np.dtype([
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

        self.key = ['u1','u2','u3','u4','u5','u6','u7','str1_bits','str2_bits']

        matches = self.read_match_data(match_data)
        idx = np.argsort(matches['u7'])
        self.dataset = matches[idx]

    def read_match_data(self,match_data):
        def parse(f):
            h5file = tables.open_file(f, mode="r")
            table = h5file.root.aln_data
            data = table.read()
            h5file.close()
            return data
        table = parse(match_data)
        return table

    def __len__(self):
        return self.dataset.shape[0]

    def span_is_valid(self,span,lo,hi):
        return span[0] < span[1] and span[0] >= lo and span[1] <= hi

    def __getitem__(self, idx):
        data = self.dataset[idx]
        q_idx = data['u1']
        t_idx = data['u2']

        query_seq = self.sequences[q_idx, :]
        target_seq = self.sequences[t_idx, :]
        query_len = self.seqlen[q_idx]
        target_len = self.seqlen[t_idx]

        for span in [[data['u3'],data['u4'],query_len],[data['u5'],data['u6'], target_len]]:
            if not self.span_is_valid(span,lo=self.offset,hi=span[2]-1+self.offset):
                print(span)
            assert(self.span_is_valid(span,lo=self.offset,hi=span[2]-1+self.offset))

        qspan = [data['u3']-self.offset,data['u4']-self.offset]
        tspan = [data['u5']-self.offset,data['u6']-self.offset]
        span = jnp.array([qspan, tspan], dtype=jnp.uint32)

        qaln = np.unpackbits(data['str1_bits'])
        taln = np.unpackbits(data['str2_bits'])
        assert(np.sum(qaln) == np.sum(taln))
        aln = jnp.array([qaln, taln], dtype=bool)

        score = jnp.array([data['u7']], dtype=jnp.float32)

        return jnp.stack([query_seq, target_seq], axis=0).astype(jnp.uint8), jnp.array([query_len, target_len], dtype=jnp.float32), span, aln, score, idx

    def export(self, outfile):
        with h5py.File(outfile, 'w') as file:
            file.create_dataset('encoding', data=self.sequences)
            file.create_dataset('seqlen', data=self.seqlen)

class TestDataset:
    def __init__(self, query_fasta, target_fasta, h5file=None, max_length=max_length):
        qdict = SeqIO.index(query_fasta,'fasta')
        tdict = SeqIO.index(target_fasta,'fasta')
        self.qsequences, self.qseqlen = create_encoding_with_len(qdict , max_length)
        self.tsequences, self.tseqlen = create_encoding_with_len(tdict , max_length)
        self.qheaders = list(qdict.keys())
        self.theaders = list(tdict.keys())

    def __len__(self):
        return min(len(self.qsequences),len(self.tsequences))

    def __getitem__(self, idxs):
        q_idx, t_idx = idxs

        query_seq = self.qsequences[q_idx, :]
        target_seq = self.tsequences[t_idx, :]
        query_len = self.qseqlen[q_idx]
        target_len = self.tseqlen[t_idx]

        return np.stack([query_seq, target_seq], axis=0).astype(np.uint8), np.array([query_len, target_len], dtype=np.float32), (self.qheaders[q_idx], self.theaders[t_idx])

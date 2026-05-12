"""
Utility functions for the project.
"""

import numpy as np
from constants import TOKENS, PAD_TOKEN, AMINO_ACIDS_GTTL, PAD_TOKEN_GTTL

tokens = AMINO_ACIDS_GTTL
pad_token = PAD_TOKEN_GTTL

def one_hot_encode_sequence(
        sequence: str,
        max_length:int = 164
) -> np.ndarray:
    """One-hot encode a protein sequence"""
    sequence = sequence.upper()
    sequence = sequence + PAD_TOKEN * (max_length - len(sequence))
    one_hot = np.zeros((max_length, len(TOKENS)))
    for i, aa in enumerate(sequence):
        try:
            one_hot[i, TOKENS.index(aa)] = 1.
        except ValueError as error:
            print(aa)
            raise error
    return one_hot

def decode_one_hot_sequence(one_hot_sequence: np.ndarray) -> str:
    """Decode a one-hot encoded to protein sequence"""
    return "".join([TOKENS[np.argmax(i)] for i in one_hot_sequence])

def create_encoding_one_hot(fasta_dict,max_length):
    fasta_dict = {k: v for i, (k, v) in enumerate(fasta_dict.items()) if i < len(fasta_dict)/4}
    encoding = np.full((len(fasta_dict),max_length,len(TOKENS)),False,dtype=bool)
    for i, seq in enumerate(fasta_dict.values()):
        padded_seq = seq + PAD_TOKEN * (max_length - len(seq))
        for j, aa in enumerate(padded_seq):
            encoding[i,j,TOKENS.index(aa)] = True
    return encoding

def create_encoding(fasta_dict,max_length):
    n = int(len(fasta_dict))
    encoding = np.empty((n,max_length),dtype=np.uint8)
    for i,seq_id in enumerate(fasta_dict.keys()):
        seq = str(fasta_dict[seq_id].seq)
        pad_idx = TOKENS.index(PAD_TOKEN)
        for j, aa in enumerate(seq):
            encoding[i,j] = TOKENS.index(aa)
        for j in range(len(seq),max_length):
            encoding[i,j] = pad_idx
    return encoding

def create_encoding_with_len(fasta_dict,max_length):
    n = int(len(fasta_dict))
    encoding = np.empty((n,max_length),dtype=np.uint8)
    seq_len = np.empty((n),dtype=np.uint16)
    for i,seq_id in enumerate(fasta_dict.keys()):
        seq = str(fasta_dict[seq_id].seq)
        seq_len[i] = len(seq)
        pad_idx = tokens.index(pad_token)
        for j, aa in enumerate(seq):
            encoding[i,j] = tokens.index(aa)
        for j in range(len(seq),max_length):
            encoding[i,j] = pad_idx
    return encoding, seq_len

import os
from pathlib import Path
import unittest
from dataset import ColBertDatasetAln
from constants import max_length, TOKENS, PAD_TOKEN
import random

INPUT_POSITIVE = '/scratch/stud2018/ata/ma_data/dataproc_bit_positive.h5'
INPUT_NEGATIVE = '/scratch/stud2018/ata/ma_data/dataproc_bit_negative.h5'
INPUT_FASTA = '/scratch/stud2018/ata/ma_data/dataproc_bit.fasta'
OUTPUT_CACHED = '/scratch/stud2018/ata/ma_data/cached_test.h5'
REFERENCE_CACHED = '/scratch/stud2018/ata/ma_data/cached_length.h5'

class TestLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = ColBertDatasetAln(
            INPUT_FASTA,
            INPUT_POSITIVE,
            INPUT_NEGATIVE,
            h5file=OUTPUT_CACHED,
            max_length=max_length
        )

    @classmethod
    def tearDownClass(cls):
        assert(os.path.isfile(OUTPUT_CACHED))
        os.remove(Path(OUTPUT_CACHED))

    def test_sequence(self):
        print('Testing sequence')
        reference_fasta = SeqIO.index(INPUT_FASTA,'fasta')
        for data in self.loader.dataset:
            q_idx = data['u1']
            p_idx = data['u2']
            n_idx = data['u7']
            for seq_idx in [q_idx, p_idx, n_idx]:
                seq_len = self.loader.seq_len[seq_idx]
                encoded_seq = self.loader.sequences[seq_idx,:]
                ref_seq = str(reference_fasta[seq_idx].seq)
                self.assertEqual(len(ref_seq),seq_len)
                self.assertFalse(TOKENS.index(PAD_TOKEN) in list(encoded_seq[0:seq_len]))
                decoded_seq = ''.join([TOKENS[s] for i,s in enumerate(encoded_seq) and i < seq_len])
                self.assertTrue(decoded_seq == ref_seq)

    def test_cache(self):
        print('Testing cache')
        loader_cached = ColBertDatasetAln(
            INPUT_FASTA,
            INPUT_POSITIVE,
            INPUT_NEGATIVE,
            h5file=REFERENCE_CACHED,
            max_length=max_length
        )
        self.assertTrue((loader.sequences == loader_cached.sequences).all())
        self.assertTrue((loader.seq_len == loader_cached.seq_len).all())

    def test_entry(self):
        print('Testing entry')
        num_of_samples = 10000
        test_indices = random.sample(range(len(self.loader)), num_of_samples)
        for i in test_indices:
            data = self.loader.dataset[i]
            q_idx = data['u1']
            p_idx = data['u2']
            n_idx = data['u7']
            seq, seqlen, _, _, alnref, idx = self.loader[i]
            self.assertEqual(idx, i)
            for j, seq_idx in enumerate([q_idx, p_idx, n_idx]):
                self.assertEqual(seqlen[j], self.loader.seq_len[seq_idx])
                self.assertTrue((seq[j] == self.loader.sequences).all())

if __name__ == "__main__":
    unittest.main()

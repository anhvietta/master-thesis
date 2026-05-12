# Content
A collection of scripts used to generate training data or to benchmark and evaluate ProtSearch

# Requirements
## MMseqs2 and MMseqs2-GPU
Install MMseqs2 and MMseqs2-GPU (https://github.com/soedinglab/mmseqs2)
## DIAMOND
Install DIAMOND (https://github.com/bbuchfink/diamond)
## Smith-Waterman implementation from GTTL
Get GTTL (https://github.com/stefan-kurtz/gttl)
## CUDASW4++
Get CUDASW4++ (https://github.com/asbschmidt/CUDASW4)

# Preamble
## Generate SW reference dataset

    gttl/tools/swalign/sw_all_against_all.x -d data/target.fasta -q data/query.fasta -v 2 -c 47 -t 6 -a 4+512 -o swref/out

# List of scripts
## Generate training data set
- sequence_matching_db.py: Convert alignment files generated using MMseqs2 to binary tabular format

## Generate results and some statistics (precision/recall)
- benchmark_pr.py and benchmark_pr_ivfpq.py: Run ProtSearchFlat and ProtSearchIVFPQ on various configs, caching result and compute precision/recall
- gttl_parser.py: Combine output files of GTTL SW to one tsv-format file
- gttl_sequence_matching_db.py: Convert alignment files generated using gttl_parser.py to binary tabular format
- score_histogram: Plot a score histogram of a GTTL SW reference and threshold the bitscore
- mmseqs_vs_sw, segsearchflat_vs_sw, segsearchivfpq_vs_sw: Compare outfile data to reference to compute precision/recall

## Plot
- diag_pr.py: Plot precision/recall distribution of aligned residues
- exhaustive_vs_approximative: Plot comparison between ProtSearchFlat and ProtSearchIVFPQ on run time
- false_positives: Compare false positives and true positives on some statistics
- index_time.py: Probe indexing (and querying) time of ProtSearchIVFPQ with various configurations
- make_hist.py: Make a histogram for diagonal recovery data
- plot_heatmap.py: Plot heatmaps from cosine similarity and reference matrices for some samples
- plot_pr.py: Plot PR curve for ProtSearch, MMseq2 and DIAMOND
- plot_runtime: Plot run time by varying a parameters while keeping other fixed
- plot_training: Plot training statistics
- runtime_comp: Plot run time comparison with MMseqs2-GPU and CUDASW4++
- ss_vs_sw_flat.py: Compute some segment- and residue-level alignment statistics
- test_ml.py: Use a simple logistic regression to discriminate false positives from true positives
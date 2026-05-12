# ProtSearch
ProtSearch is a workflow written in C++ that prefilters a query sequence set Q against a target sequence set T and approximates their alignments by embedding each query amino acid in a vector space.

# Requirements
## C++
Faiss
LibTorch
CUDA
BLAS
## Python
PyTorch

# Usage
## Compiling

## Make db

    makedb.x -m ckpts/ckpts.pt --export_index index data/target.fasta
## Query

    query.x -m ckpts/ckpts.pt -i index data/target.fasta data/query.fasta

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
#import umap
from utils import tokens
from pathlib import Path
import torch

base_path = "/scratch/stud2018/ata/"

def load_snapshots():
    snapshot_path = Path(base_path)
    snapshot_files = snapshot_path.glob("ckpts_pyramid_ptwise_noskip_120m_0.07_32+96+160+256+320+384+480_3+5+7+9+11+15+17_1+2+3+4+5+6+7_320_0+2+4+6_tokenizer_*")
    snapshots = []
    for file in snapshot_files:
        embedding = torch.load(file.resolve()).numpy()
        snapshots.append(embedding)
    return snapshots
'''
snapshots = load_snapshots()
# Stack all embeddings to fit PCA once
#all_embeddings = np.vstack(snapshots)
aligned = [snapshots[0]]  # reference

for i in range(1, len(snapshots)):
    R, _ = orthogonal_procrustes(snapshots[i], aligned[i-1])
    aligned.append(snapshots[i] @ R)
all_embeddings = np.vstack(aligned)

# Fit PCA
pca = PCA(n_components=2)
pca.fit(all_embeddings)

# Project each snapshot
projected = [pca.transform(E) for E in snapshots]

# Example: plot a few tokens over time
token_ids = [i for i in range(20)]  # choose tokens to track

plt.figure(figsize=(6, 6))

for token_id in token_ids:
    traj = np.array([proj[token_id] for proj in projected])
    plt.plot(traj[:, 0], traj[:, 1], marker='o', label=tokens[token_id])

plt.title("Token trajectories (PCA)")
plt.legend()
#plt.show()
plt.savefig(base_path + "measurements2/pca_visualize.png")

# Fit UMAP once
reducer = umap.UMAP(n_components=2, random_state=42)
reducer.fit(all_embeddings)

# Transform each snapshot
projected = [reducer.transform(E) for E in snapshots]

# Plot trajectories
token_ids = [i for i in range(20)]  # choose tokens to track

plt.figure(figsize=(6, 6))

for token_id in token_ids:
    traj = np.array([proj[token_id] for proj in projected])
    plt.plot(traj[:, 0], traj[:, 1], marker='o', label=tokens[token_id])

plt.title("Token trajectories (UMAP)")
plt.legend()
#plt.show()
plt.savefig(base_path + "measurements2/umap_visualize.png")
'''
last_snapshot = 'ckpts_pyramid_ptwise_noskip_120m_0.07_32+96+160+256+320+384+480_3+5+7+9+11+15+17_1+2+3+4+5+6+7_320_0+2+4+6_tokenizer_480000.pth'
embedding = torch.load(last_snapshot).numpy()
pca_embedding = PCA(n_components=2).fit_transform(embedding)
'''fig = plt.figure()
ax = fig.add_subplot()
assert(len(pca_embedding) == len(tokens))
for emb, aa in zip(pca_embedding, tokens):
    ax.scatter(emb[0], emb[1])
    ax.text(emb[0], emb[1], aa, None)

ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
#plt.show()
plt.savefig(base_path + "measurements2/pca_token.png")
'''
from constants import blosum62_gttl
K = embedding @ embedding.T
b = np.array(blosum62_gttl, dtype=np.float32)
from scipy.stats import spearmanr

def spearman_matrix_correlation(A, B):
    """
    Compute Spearman correlation between two matrices.

    Parameters:
        A, B: numpy arrays of the same shape

    Returns:
        correlation, p_value
    """
    if A.shape != B.shape:
        raise ValueError("Matrices must have the same shape.")

    # Flatten matrices to 1D
    A_flat = A.flatten()
    B_flat = B.flatten()

    # Compute Spearman correlation
    corr, p_value = spearmanr(A_flat, B_flat)

    return corr, p_value
print(K.shape, b.shape)
assert(K.shape == b.shape)
corr, p = spearman_matrix_correlation(K, b)

print("Spearman correlation:", corr)
print("p-value:", p)

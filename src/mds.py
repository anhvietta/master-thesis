import numpy as np
from constants import blosum62_raw, blosum62_gttl, AMINO_ACIDS_GTTL
from sklearn import manifold, decomposition
import matplotlib.pyplot as plt


def mds_from_similarity(S, d=None):
    """
    Classical MDS from a similarity (Gram) matrix.

    Args:
        S (ndarray): similarity matrix (n x n, symmetric, ideally PSD).
        d (int): target dimension. If None, uses full rank.

    Returns:
        V (ndarray): reconstructed vectors (n x d).
        S_approx (ndarray): approximated similarity matrix (n x n).
        eigvals (ndarray): eigenvalues sorted descending.
    """
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(S)  # eigh since symmetric
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    if d is None:
        d = (eigvals > 1e-12).sum()  # effective rank

    # Keep top d
    L_d = np.diag(np.sqrt(np.maximum(eigvals[:d], 0)))
    Q_d = eigvecs[:, :d]

    V = Q_d @ L_d
    S_approx = V @ V.T

    return V, S_approx, eigvals


def min_dimension_for_error(S, tol=0.1):
    """
    Find smallest dimension d such that
    max absolute error between S and reconstruction <= tol.
    """
    n = S.shape[0]
    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    for d in range(1, n+1):
        L_d = np.diag(np.sqrt(np.maximum(eigvals[:d], 0)))
        Q_d = eigvecs[:, :d]
        V = Q_d @ L_d
        S_approx = V @ V.T
        err = np.max(np.abs( (S - S_approx) * (S > 0) ))
        if err <= tol:
            return d, err

    return n, err  # full dimension needed

def psd_projection(S, tol=0.0):
    S = (S + S.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals_clipped = np.clip(eigvals, a_min=tol, a_max=None)
    S_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return S_psd, eigvals, eigvals_clipped

def psd_truncated_embedding(S, d):
    S_psd, eigvals, eigvals_clipped = psd_projection(S)
    eigvals_pos, eigvecs = np.linalg.eigh(S_psd)
    idx = np.argsort(eigvals_pos)[::-1]
    eigvals_pos = eigvals_pos[idx]
    eigvecs = eigvecs[:, idx]
    Ld = np.diag(np.sqrt(eigvals_pos[:d]))
    Qd = eigvecs[:, :d]
    V = Qd @ Ld
    return V, V @ V.T

def get_mdsdecmp():
    blosum62 = np.array(blosum62_gttl)[:-1,:-1]
    V, _, _ = mds_from_similarity(blosum62, d=12)
    return V

if __name__ == '__main__':
    blosum62 = np.array(blosum62_gttl)[:-4,:-4]
    '''n, err = min_dimension_for_error(blosum62, 2.09)
    print(n, err)
    V, S_approx, eigvals = mds_from_similarity(blosum62, d=12)
    print(V, np.max(np.abs((blosum62 - S_approx) * (blosum62 > 0))), eigvals)
    print(S_approx)'''
    embedding = manifold.SpectralEmbedding(n_components=18, n_neighbors=4)
    le = embedding.fit_transform(blosum62-blosum62.min())
    print(le.shape)
    pca = decomposition.PCA(n_components=2).fit_transform(le)
    fig = plt.figure()
    ax = fig.add_subplot()
    for emb, aa in zip(pca, AMINO_ACIDS_GTTL):
        ax.scatter(emb[0], emb[1])
        ax.text(emb[0], emb[1], aa, None)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    #ax.set_zlabel('TSNE3')
    #plt.show()
    plt.savefig("/scratch/stud2018/ata/le.png")

from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from prismg.utils.grm import standardize_by_af, filter_monomorphic
from prismg.score import clamp01

def fit_pca(X, n_components: int = 10, random_state: int = 123, randomized: bool = True):
    X = np.asarray(X, dtype=np.float32)
    p = PCA(
        n_components=n_components,
        random_state=random_state,
        svd_solver=("randomized" if randomized else "auto"))
    return p, p.fit_transform(X)

def project(pca: PCA, X: np.ndarray) -> np.ndarray:
    return pca.transform(np.asarray(X, dtype=np.float32))

def nn_min_dists(X_query: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_ref)
    d, _ = nn.kneighbors(X_query, n_neighbors=1, return_distance=True)
    return d[:, 0]

def nn_real_baseline(X_ho: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(X_ho)
    dists, _ = nn.kneighbors(X_ho, n_neighbors=2)
    return dists[:, 1]

def quantile(x, q: float) -> float:
    return float(np.quantile(np.asarray(x), q))

def pli_metrics(X_ho: np.ndarray, X_syn: np.ndarray, q: float = 0.01) -> Dict[str, object]:
    # Distances per hold-out
    d_syn = nn_min_dists(X_ho, X_syn)
    d_real = nn_real_baseline(X_ho)

    # Quantile ratio
    rho_q = quantile(d_syn, q) / max(quantile(d_real, q), 1e-12)
    r_p = clamp01(1.0 - rho_q)

    # Adversarial proximity test
    A = float(np.mean(d_real < d_syn))
    r_A = float(max(0.0, 0.5 - A) / 0.5)

    pli_val = float(max(r_p, r_A))

    return {
        "d_syn": d_syn,
        "d_real": d_real,
        "rho_q": float(rho_q),
        "r_p": r_p,
        "A": A,
        "r_A": r_A,
        "PLI": pli_val,
    }
    
def compute_pli(G_tr: np.ndarray, G_ho: np.ndarray, G_syn: np.ndarray, n_components: int = 10, random_state: int = 123, q: float = 0.01, *, randomized_pca: bool = True, use_filter_monomorphic: bool = True, mono_tol: float = 1e-8) -> Dict[str, object]:
    # Standardize using TRAIN allele frequencies
    G_tr_std, _  = standardize_by_af(G_tr, G_tr, dtype=np.float32)
    if use_filter_monomorphic:
        G_tr_std, keep = filter_monomorphic(G_tr_std, tol=mono_tol)
    else:
        keep = slice(None)  # no filtering

    G_ho_std, _  = standardize_by_af(G_ho, G_tr, dtype=np.float32)
    G_syn_std, _ = standardize_by_af(G_syn, G_tr, dtype=np.float32)

    G_ho_std  = G_ho_std[:, keep]
    G_syn_std = G_syn_std[:, keep]

    # PCA fit on TRAIN, project others
    pca, X_tr = fit_pca(G_tr_std, n_components=n_components, random_state=random_state, randomized=randomized_pca)
    X_ho  = project(pca, G_ho_std)
    X_syn = project(pca, G_syn_std)

    out = pli_metrics(X_ho, X_syn, q=q)
    out.update({
        "n_components": int(n_components),
        "q": float(q),
        "filtered_monomorphic": bool(use_filter_monomorphic),
        "kept_snps": (G_tr_std.shape[1] if isinstance(keep, slice) else int(np.sum(keep))),
    })
    return out


__all__ = [
    "fit_pca",
    "project",
    "nn_min_dists",
    "nn_real_baseline",
    "quantile",
    "pli_index",
    "compute_pli",
]
"""
PLI — Proximity Leakage Index 

Implements the definition:
- Nearest synthetic distance vs. nearest real distance
- Quantile ratio rho_q
- Proximity risk signal r_p
- Adversarial proximity accuracy A and risk r_A
- Final PLI = max(r_p, r_A)

Usage pattern A (projected coords):
-----------------------------------
pca, X_train = fit_pca(G_train, n_components=10)
X_real  = project(pca, G_real)
X_ho    = project(pca, G_ho)
X_model = project(pca, G_model)
results = pli_index(X_ho, X_real, X_model)
print(results)

Usage pattern B (raw genotypes → one-shot):
-------------------------------------------
out = compute_pli(G_tr, G_ho, G_syn, n_components=10, random_state=42, q=0.01)
print(out)
"""
from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Optional high-level preprocessing
from prism_g.grm import standardize_by_af, filter_monomorphic


# ==================== PCA utils (optimized) ====================

def fit_pca(X, n_components: int = 10, random_state: int = 42, randomized: bool = True):
    """
    Fit PCA on standardized data (float32). Uses randomized SVD by default for speed.
    """
    X = np.asarray(X, dtype=np.float32)
    p = PCA(
        n_components=n_components,
        random_state=random_state,
        svd_solver=("randomized" if randomized else "auto"),
    )
    return p, p.fit_transform(X)


def project(pca: PCA, X: np.ndarray) -> np.ndarray:
    """
    Project data with a fitted PCA object (casts to float32).
    """
    return pca.transform(np.asarray(X, dtype=np.float32))


# ==================== Distance helpers ====================

def nn_min_dists(X_query: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_ref)
    d, _ = nn.kneighbors(X_query, n_neighbors=1, return_distance=True)
    return d[:, 0]


def nearest_real_baseline(X_ho: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(X_ho)
    dists, _ = nn.kneighbors(X_ho, n_neighbors=2)
    return dists[:, 1]


def quantile(x, q: float) -> float:
    return float(np.quantile(np.asarray(x), q))


# ==================== PLI index ====================

def pli_index(
    X_ho: np.ndarray,
    X_real: np.ndarray,
    X_syn: np.ndarray,
    q: float = 0.01,
) -> Dict[str, object]:
    """Compute the Proximity Leakage Index and return all components."""
    # Distances per hold-out
    d_syn = nn_min_dists(X_ho, X_syn)
    d_real = nearest_real_baseline(X_ho)

    # Quantile ratio
    rho_q = quantile(d_syn, q) / max(quantile(d_real, q), 1e-12)
    r_p = float(np.clip(1.0 - rho_q, 0.0, 1.0))

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


# ==================== High-level convenience ====================

def compute_pli(
    G_tr: np.ndarray,
    G_ho: np.ndarray,
    G_syn: np.ndarray,
    n_components: int = 10,
    random_state: int = 42,
    q: float = 0.01,
    *,
    randomized_pca: bool = True,
    use_filter_monomorphic: bool = True,
    mono_tol: float = 1e-8,
) -> Dict[str, object]:
    """
    End-to-end PLI from raw genotypes.

    Steps:
      TRAIN-AF standardize (float32) →
      optional monomorphic filtering (based on TRAIN) →
      PCA on TRAIN → project HO & SYN → PLI components.

    Returns same dictionary as `pli_index` plus metadata keys.
    """
    # Standardize using TRAIN allele frequencies
    G_tr_std, _  = standardize_by_af(G_tr, G_tr, dtype=np.float32)
    if use_filter_monomorphic:
        G_tr_std, keep = filter_monomorphic(G_tr_std, tol=mono_tol)
    else:
        keep = slice(None)  # no filtering

    G_ho_std, _  = standardize_by_af(G_ho, G_tr, dtype=np.float32)
    G_syn_std, _ = standardize_by_af(G_syn, G_tr, dtype=np.float32)

    # Apply same column mask to keep alignment
    G_ho_std  = G_ho_std[:, keep]
    G_syn_std = G_syn_std[:, keep]

    # PCA fit on TRAIN, project others
    pca, X_tr = fit_pca(G_tr_std, n_components=n_components, random_state=random_state, randomized=randomized_pca)
    X_ho  = project(pca, G_ho_std)
    X_syn = project(pca, G_syn_std)

    # Compute PLI on projected space
    out = pli_index(X_ho, X_tr, X_syn, q=q)
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
    "nearest_real_baseline",
    "quantile",
    "pli_index",
    "compute_pli",
]

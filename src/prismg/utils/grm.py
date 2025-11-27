from __future__ import annotations

from typing import Tuple
import numpy as np

def _symmetrize(K: np.ndarray) -> np.ndarray:
    return 0.5 * (K + K.T)

def _af_from_ref(G_ref: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    G = np.asarray(G_ref, dtype=float)
    col_means = np.nanmean(G, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    p = np.clip(col_means / 2.0, eps, 1.0 - eps)
    return p

def standardize_by_af(G: np.ndarray, G_ref: np.ndarray, eps: float = 1e-6,*,dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    # compute p from reference (float64 for stability in AF)
    p = _af_from_ref(np.asarray(G_ref, dtype=float), eps=eps)
    denom = np.sqrt(2.0 * p * (1.0 - p)) + eps

    # standardize target in chosen dtype
    G = np.asarray(G, dtype=np.float32 if dtype == np.float32 else np.float64)
    G_imp = np.where(np.isnan(G), 2.0 * p, G)
    X = (G_imp - 2.0 * p) / denom
    X = X.astype(dtype, copy=False)
    return X, p

def filter_monomorphic(X: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X)
    keep = np.nanvar(X, axis=0) > tol
    return X[:, keep], keep

def compute_grm_blocked(X: np.ndarray, block_cols: int = 8192) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n, m = X.shape
    if m == 0:
        return np.zeros((n, n), dtype=np.float32)

    K = np.zeros((n, n), dtype=np.float32)
    for j0 in range(0, m, block_cols):
        j1 = min(j0 + block_cols, m)
        B = X[:, j0:j1]  # n x b
        K += B @ B.T     # BLAS, float32
    K /= float(m)
    return _symmetrize(K)

def compute_grm(X: np.ndarray) -> np.ndarray:
    return compute_grm_blocked(X, block_cols=8192)

__all__ = [
    "standardize_by_af",
    "filter_monomorphic",
    "compute_grm_blocked",
    "compute_grm",
]
"""
GRM utilities for PRISM-G 

Includes:
- AF-based standardization with NaN→2p imputation (float32-friendly)
- Monomorphic SNP filtering
- Blocked GRM computation to reduce memory and speed up large m 

Conventions
-----------
- G and G_ref are numpy arrays (n_samples, n_snps) with genotypes in {0,1,2} and NaN.
- Allele frequency p is computed from the reference matrix only (column-wise mean/2).
- Standardization uses X = (G_imp - 2p) / sqrt(2p(1-p)), with epsilon guards.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np


# =============================
# Helpers
# =============================

def _symmetrize(K: np.ndarray) -> np.ndarray:
    return 0.5 * (K + K.T)


def _af_from_ref(G_ref: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    G = np.asarray(G_ref, dtype=float)
    col_means = np.nanmean(G, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    p = np.clip(col_means / 2.0, eps, 1.0 - eps)
    return p


# =============================
# Core functions (optimized)
# =============================

def standardize_by_af(
    G: np.ndarray,
    G_ref: np.ndarray,
    eps: float = 1e-6,
    *,
    dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """AF-based standardization with NaN→2p imputation.

    Parameters
    ----------
    G : array-like (n, m)
        Target genotype matrix to standardize.
    G_ref : array-like (n_ref, m)
        Reference matrix used to estimate allele frequencies.
    eps : float
        Numerical guard for extremes in p and denominators.
    dtype : numpy dtype
        Output dtype for X; float32 recommended for speed/memory.

    Returns
    -------
    X : standardized matrix (n, m) with dtype `dtype`
    p : allele frequencies (m,)
    """
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
    """Drop ~monomorphic SNPs by variance threshold.

    Returns
    -------
    X_keep : (n, k) with columns filtered
    keep_mask : (m,) boolean mask of kept columns
    """
    X = np.asarray(X)
    keep = np.nanvar(X, axis=0) > tol
    return X[:, keep], keep


def compute_grm_blocked(X: np.ndarray, block_cols: int = 8192) -> np.ndarray:
    """Compute K = (X X^T) / m in column blocks (fast, low memory).

    Notes
    -----
    - X should already be standardized; float32 is recommended.
    - The computation accumulates over blocks of columns to avoid large temporaries.
    """
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
    """Backward-compatible wrapper using the blocked kernel by default."""
    return compute_grm_blocked(X, block_cols=8192)


__all__ = [
    "standardize_by_af",
    "filter_monomorphic",
    "compute_grm_blocked",
    "compute_grm",
]
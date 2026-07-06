from __future__ import annotations

from typing import Tuple
import numpy as np

def _symmetrize(K: np.ndarray) -> np.ndarray:
    """Force exact symmetry in a square matrix by averaging with its transpose.
 
    Parameters
    ----------
    K : np.ndarray
        Square matrix.
 
    Returns
    -------
    np.ndarray
        ``0.5 * (K + K.T)``.
    """
    return 0.5 * (K + K.T)

def _af_from_ref(G_ref: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Estimate per-SNP ALT allele frequencies from a reference dosage matrix.
 
    Parameters
    ----------
    G_ref : np.ndarray
        Reference dosage matrix.
    eps : float, default=1e-6
        Clip bound applied after dividing.
 
    Returns
    -------
    np.ndarray
        Estimated ALT allele frequency per SNP.
    """
    G = np.asarray(G_ref, dtype=float)
    col_means = np.nanmean(G, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    p = np.clip(col_means / 2.0, eps, 1.0 - eps)
    return p

def standardize_by_af(G: np.ndarray, G_ref: np.ndarray, eps: float = 1e-6,*,dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """Centre and scale a dosage matrix by reference-panel allele frequencies.
 
    Parameters
    ----------
    G : np.ndarray
        Genotypoe dosage matrix.
    G_ref : np.ndarray
        Reference dosage matrix for AF estimation.
    eps : float, default=1e-6
        Clip bound for allele frequencies.
    dtype : numpy dtype, default=np.float32
        Output dtype of the standardised matrix. ``float32`` is the
        default to reduce memory for large panels; use ``float64`` where
        higher precision is needed.
 
    Returns
    -------
    X : np.ndarray
        AF-standardised matrix.
    p : np.ndarray
        Estimated ALT allele frequencies.
    """
    p = _af_from_ref(np.asarray(G_ref, dtype=float), eps=eps)
    denom = np.sqrt(2.0 * p * (1.0 - p)) + eps

    # standardize target in chosen dtype
    G = np.asarray(G, dtype=np.float32 if dtype == np.float32 else np.float64)
    G_imp = np.where(np.isnan(G), 2.0 * p, G)
    X = (G_imp - 2.0 * p) / denom
    X = X.astype(dtype, copy=False)
    return X, p

def filter_monomorphic(X: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Remove near-zero-variance columns from a standardised genotype matrix.
 
    Parameters
    ----------
    X : np.ndarray
        Standardised genotype matrix.
    tol : float, default=1e-8
        Variance threshold below which a column is considered
        monomorphic and removed.
 
    Returns
    -------
    X_filtered : np.ndarray
        Input matrix with low-variance columns removed.
    keep : np.ndarray
        Boolean mask of shape ``(m,)``, ``True`` for retained columns.
        Apply to other matrices with ``G_other[:, keep]`` to keep the
        same SNP set across cohorts.
    """
    X = np.asarray(X)
    keep = np.nanvar(X, axis=0) > tol
    return X[:, keep], keep

def compute_grm_blocked(X: np.ndarray, block_cols: int = 8192) -> np.ndarray:
    """Compute a Genomic Relationship Matrix (GRM) in memory-bounded blocks.
 
    Parameters
    ----------
    X : np.ndarray
        AF-standardised genotype matrix.
    block_cols : int, default=8192
        Number of SNP columns per block. Larger values use more memory
        but may be faster due to better BLAS utilisation. Reduce for
        very large ``m`` or limited RAM.
 
    Returns
    -------
    np.ndarray
        Symmetric GRM.
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
    """Compute a GRM using the default block size.
 
    Parameters
    ----------
    X : np.ndarray
        AF-standardised genotype matrix.
 
    Returns
    -------
    np.ndarray
        Symmetric GRM of shape.
    """
    return compute_grm_blocked(X, block_cols=8192)

__all__ = [
    "standardize_by_af",
    "filter_monomorphic",
    "compute_grm_blocked",
    "compute_grm",
]
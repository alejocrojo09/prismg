from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from prismg.utils.grm import standardize_by_af, filter_monomorphic
from score import clamp01, _aggregate

def fit_pca(X, n_components: int = 10, random_seed: int = 123, randomized: bool = True):
    """Fit a PCA model and return both the fitted model and the projected data.
 
    Parameters
    ----------
    X : np.ndarray
        Genotype matrix.
    n_components : int, default=10
        Number of principal components to retain.
    random_seed : int, default=123
        Random seed.
    randomized : bool, default=True
        If ``True``, uses the randomized SVD solver. If ``False``, uses scikit-learn's ``"auto"`` solver
        selection.
 
    Returns
    -------
    pca : sklearn.decomposition.PCA
        The fitted PCA object. Pass to ``project()`` to transform
        holdout or synthetic matrices into the same PC space.
    X_tr : np.ndarray
        Training data projected into PC space.
    """
    X = np.asarray(X, dtype=np.float32)
    p = PCA(
        n_components=n_components,
        random_state=random_seed,
        svd_solver=("randomized" if randomized else "auto"))
    return p, p.fit_transform(X)

def project(pca: PCA, X: np.ndarray) -> np.ndarray:
    """Project a genotype matrix into an already-fitted PCA space.
 
    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        A PCA object previously fitted via ``fit_pca``.
    X : np.ndarray
        Matrix to project.
 
    Returns
    -------
    np.ndarray
        Projected matrix.
    """
    return pca.transform(np.asarray(X, dtype=np.float32))

def nn_min_dists(X_query: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    """Compute the Euclidean distance from each query point to its nearest reference point.
 
    Parameters
    ----------
    X_query : np.ndarray
        Query points. Each row is one
        individual's PC coordinates (e.g. holdout cohort).
    X_ref : np.ndarray
        Reference points. Each row is one
        individual's PC coordinates (e.g. synthetic cohort).
 
    Returns
    -------
    np.ndarray
        1D array containing the distance from
        each query point to its single nearest neighbour in ``X_ref``.
    """
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_ref)
    d, _ = nn.kneighbors(X_query, n_neighbors=1, return_distance=True)
    return d[:, 0]

def nn_real_baseline(X_ho: np.ndarray) -> np.ndarray:
    """Compute each holdout individual's distance to their nearest *other* holdout individual.
 
    Parameters
    ----------
    X_ho : np.ndarray
        Holdout cohort in PC space.
 
    Returns
    -------
    np.ndarray
        1D array containing the distance from each
        holdout individual to their nearest *other* holdout individual.
    """
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(X_ho)
    dists, _ = nn.kneighbors(X_ho, n_neighbors=2)
    return dists[:, 1]

def quantile(x, q: float) -> float:
    """Return the q-th quantile of an array.
 
    Parameters
    ----------
    x : array-like
        Input values.
    q : float
        Quantile level in [0, 1].
 
    Returns
    -------
    float
        The q-th quantile of ``x``.
    """
    return float(np.quantile(np.asarray(x), q))

def _aggregate(values: list, agg: str) -> float:
    """Aggregate a list of sub-scores using max, mean, or median."""
    if agg == "mean":
        return float(np.mean(values))
    if agg == "median":
        return float(np.median(values))
    return float(max(values))  # default: max
 
 
def pli_metrics(X_ho: np.ndarray, X_syn: np.ndarray, q: float = 0.01, agg: str = "max") -> Dict[str, object]:
    """Compute PLI sub-scores and diagnostics from PC-space distance arrays.
 
    Parameters
    ----------
    X_ho : np.ndarray
        Holdout cohort in PC space.
    X_syn : np.ndarray
        Synthetic cohort in PC space.
    q : float, default=0.01
        Quantile level for the quantile-ratio sub-score. The default
        ``0.01`` focuses on the closest 1% of holdout->synthetic pairs,
        where leakage signal is strongest.
    agg : str, default="max"
        Aggregation method for combining ``r_p`` and ``r_A`` into the
        final PLI. One of "max", "mean", or "median".
 
    Returns
    -------
    dict
        Result dictionary with the following keys:
 
        - "d_syn" (np.ndarray): per-holdout-individual distance to
          the nearest synthetic neighbour.
        - "d_real" (np.ndarray): per-holdout-individual distance to
          the nearest other holdout neighbour.
        - "rho_q" (float): quantile ratio
          ``Q_q(d_syn) / Q_q(d_real)``.
        - "r_p" (float): quantile-ratio sub-score in [0, 1].
        - "A" (float): fraction of holdout individuals with a closer
          real neighbour than synthetic neighbour.
        - "r_A" (float): adversarial proximity sub-score in [0, 1].
        - "PLI" (float): final PLI in [0, 1].
    """
    # Distances per hold-out
    d_syn = nn_min_dists(X_ho, X_syn)
    d_real = nn_real_baseline(X_ho)

    # Quantile ratio
    rho_q = quantile(d_syn, q) / max(quantile(d_real, q), 1e-12)
    r_p = clamp01(1.0 - rho_q)

    # Adversarial proximity test
    A = float(np.mean(d_real < d_syn))
    r_A = float(max(0.0, 0.5 - A) / 0.5)

    pli_val = _aggregate([r_p, r_A], agg)

    return {
        "d_syn": d_syn,
        "d_real": d_real,
        "rho_q": float(rho_q),
        "r_p": r_p,
        "A": A,
        "r_A": r_A,
        "PLI": pli_val,
    }
    
def compute_pli(G_tr: np.ndarray, G_ho: np.ndarray, G_syn: np.ndarray, n_components: int = 10, random_seed: int = 123, q: float = 0.01, *, randomized_pca: bool = True, use_filter_monomorphic: bool = True, mono_tol: float = 1e-8, agg: str = "max") -> Dict[str, object]:
    """Compute the full Proximity Leakage Index (PLI) for a synthetic cohort.
 
    Parameters
    ----------
    G_tr : np.ndarray
        Training genotype matrix. Used for AF
        standardization and PCA fitting.
    G_ho : np.ndarray
        Holdout genotype matrix. Serves as the
        null reference for proximity comparisons.
    G_syn : np.ndarray
        Synthetic genotype matrix to evaluate.
    n_components : int, default=10
        Number of principal components to use.
    random_state : int, default=123
        Random seed.
    q : float, default=0.01
        Quantile level for the quantile-ratio sub-score.
    randomized_pca : bool, default=True
        Whether to use the randomized SVD solver in PCA.
    use_filter_monomorphic : bool, default=True
        If ``True``, removes SNP columns with near-zero variance in the
        (standardized) training matrix before PCA. The same column mask
        is applied to the holdout and synthetic matrices so all three
        share the same SNP set. Monomorphic SNPs carry no information in
        PCA and can introduce numerical noise.
    mono_tol : float, default=1e-8
        Variance tolerance below which a column is considered
        monomorphic. Only used
        when ``use_filter_monomorphic=True``.
    agg : str, default="max"
        Aggregation method for combining ``r_p`` and ``r_A`` into the
        final PLI, forwarded to ``pli_metrics``. One of ``"max"``,
        ``"mean"``, or ``"median"``.
 
    Returns
    -------
    dict
        All keys returned by ``pli_metrics`` ("d_syn", "d_real",
        "rho_q", "r_p", "A", "r_A", "PLI") plus:
 
        - "n_components" (int): number of PCs used.
        - "q" (float): quantile level used.
        - "filtered_monomorphic" (bool): whether monomorphic
          filtering was applied.
        - "kept_snps" (int): number of SNP columns retained after
          filtering.
    """
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
    pca, X_tr = fit_pca(G_tr_std, n_components=n_components, random_seed=random_seed, randomized=randomized_pca)
    X_ho  = project(pca, G_ho_std)
    X_syn = project(pca, G_syn_std)

    out = pli_metrics(X_ho, X_syn, q=q, agg=agg)
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
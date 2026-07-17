from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prismg.metrics.pli import compute_pli
from prismg.score import clamp01


def per_sample_pli_table(d_syn: np.ndarray, d_real: np.ndarray, sample_names: Optional[Sequence[str]] = None, eps: float = 1e-12) -> pd.DataFrame:
    """Build a per-holdout-sample PLI diagnostics table.
 
    Parameters
    ----------
    d_syn : np.ndarray
        Per-sample distance to the nearest synthetic neighbour.
    d_real : np.ndarray
        Per-sample distance to the nearest other holdout neighbour
        (the null baseline).
    sample_names : sequence of str, optional
        Identifiers for each holdout sample. If ``None``, samples are
        labelled "HO_0", "HO_1", etc.
    eps : float, default=1e-12
        Small constant added to ``d_real`` when computing the ratio
        ``d_syn / d_real``, to avoid division by zero.
 
    Returns
    -------
    pd.DataFrame
        One row per holdout sample, sorted descending by ``s_ratio``
        then ``margin_dreal_minus_dsyn``. Columns:
 
        - "sample": sample identifier.
        - "d_syn": distance to nearest synthetic neighbour.
        - "d_real": distance to nearest other holdout neighbour
          (2-NN baseline).
        - "ratio_dsyn_dreal": ``d_syn / d_real``. Values below 1
          indicate the synthetic cohort is closer than the real baseline
          (risk signal).
        - "margin_dreal_minus_dsyn": ``d_real - d_syn``. Positive
          values indicate the synthetic cohort is closer (risk signal);
          larger margins indicate stronger identifiability.
        - "s_ratio" per-sample proximity score
          ``clamp(1 - ratio, 0, 1)``. Values near 1 indicate high
          proximity risk for this individual.
        - "ap_flag": 1 if ``d_syn < d_real`` (adversarial-proximity
          candidate), else 0.
        - "rank_s_ratio": rank by ``s_ratio`` descending
          (1 = most at risk).
        - "rank_margin": rank by ``margin_dreal_minus_dsyn``
          descending (1 = largest margin).
    """
    d_syn = np.asarray(d_syn, float)
    d_real = np.asarray(d_real, float)
    if d_syn.shape != d_real.shape:
        raise ValueError("d_syn and d_real must have the same shape")
    n = len(d_syn)
    if sample_names is None:
        sample_names = [f"HO_{i}" for i in range(n)]

    ratio  = d_syn / (d_real + eps)
    margin = d_real - d_syn

    df = pd.DataFrame({
        "sample":                  list(sample_names),
        "d_syn":                   d_syn,
        "d_real":                  d_real,
        "ratio_dsyn_dreal":        ratio,
        "margin_dreal_minus_dsyn": margin,
        "s_ratio":                 np.clip(1.0 - ratio, 0.0, 1.0),
        "ap_flag":                 (d_syn < d_real).astype(int),
    })
    df["rank_s_ratio"] = (-df["s_ratio"]).rank(method="min").astype(int)
    df["rank_margin"]  = (-df["margin_dreal_minus_dsyn"]).rank(method="min").astype(int)

    return df.sort_values(
        ["s_ratio", "margin_dreal_minus_dsyn"], ascending=False
    ).reset_index(drop=True)


def bootstrap_pli_consistency(d_syn: np.ndarray, d_real: np.ndarray, q: float = 0.01, boot: int = 100, random_seed: int = 123) -> pd.DataFrame:
    """Assess PLI sub-score stability by bootstrapping holdout sample indices.

    Parameters
    ----------
    d_syn : np.ndarray
        Per-sample distance to the nearest synthetic neighbour.
    d_real : np.ndarray
        Per-sample holdout 2-NN baseline distances.
    q : float, default=0.01
        Quantile level for the ``rho_q`` / ``r_p`` sub-score.
    boot : int, default=100
        Number of bootstrap resamples.
    random_seed : int, default=123
        Random seed.
 
    Returns
    -------
    pd.DataFrame
        One row per bootstrap resample, ``boot`` rows total. Columns:
        ``rho_q``, ``r_p``, ``A``, ``r_A``, ``PLI`` -- the same five
        scalar outputs as ``pli_metrics``. Summarise with
        ``.describe()`` or ``.quantile([0.025, 0.975])`` to obtain
        confidence intervals.
    """
    rng   = np.random.default_rng(random_seed)
    d_syn  = np.asarray(d_syn, float)
    d_real = np.asarray(d_real, float)
    n     = len(d_syn)

    rows = []
    for _ in range(boot):
        idx = rng.integers(0, n, size=n)
        ds, dr = d_syn[idx], d_real[idx]

        rho_q = float(np.quantile(ds, q)) / max(float(np.quantile(dr, q)), 1e-12)
        r_p   = clamp01(1.0 - rho_q)
        A     = float(np.mean(dr < ds))
        r_A   = float(max(0.0, 0.5 - A) / 0.5)

        rows.append({"rho_q": rho_q, "r_p": r_p, "A": A, "r_A": r_A, "PLI": max(r_p, r_A)})

    return pd.DataFrame(rows)


def scan_pli_stability(G_tr: np.ndarray, G_ho: np.ndarray, G_syn: np.ndarray, *, n_components_grid: Iterable[int] = (5, 10, 20), q_grid: Iterable[float] = (0.005, 0.01, 0.02), seeds: Iterable[int] = (123, 124, 125, 126, 127), randomized_pca: bool = True) -> pd.DataFrame:
    """Recompute the full PLI pipeline across a grid of hyperparameters and seeds.
 
    Parameters
    ----------
    G_tr : np.ndarray
        Training genotype matrix.
    G_ho : np.ndarray
        Holdout genotype matrix,
    G_syn : np.ndarray
        Synthetic genotype matrix.
    n_components_grid : iterable int, default=(5, 10, 20)
        PCA component counts to sweep over.
    q_grid : iterable of float, default=(0.005, 0.01, 0.02)
        Quantile levels for the ``r_p`` sub-score to sweep over.
    seeds : iterable of int, default=(123, 124, 125, 126, 127)
        Random seeds for PCA to sweep over. Multiple seeds reveal
        variability from PCA's randomized SVD solver.
    randomized_pca : bool, default=True
        Whether to use the randomized SVD solver in PCA, forwarded to
        ``compute_pli``.
 
    Returns
    -------
    pd.DataFrame
        One row per (``n_components``, ``q``, ``seed``) combination.
        Columns: ``n_components``, ``q``, ``seed``, ``rho_q``, ``r_p``,
        ``A``, ``r_A``, ``PLI``, ``kept_snps``.
        Total rows: ``len(n_components_grid) * len(q_grid) * len(seeds)``.
    """
    rows = []
    for nc in n_components_grid:
        for q in q_grid:
            for s in seeds:
                out = compute_pli(
                    G_tr, G_ho, G_syn,
                    n_components=nc,
                    random_seed=s,
                    q=q,
                    randomized_pca=randomized_pca,
                )
                rows.append({
                    "n_components": nc,
                    "q":            q,
                    "seed":         s,
                    "rho_q":        out["rho_q"],
                    "r_p":          out["r_p"],
                    "A":            out["A"],
                    "r_A":          out["r_A"],
                    "PLI":          out["PLI"],
                    "kept_snps":    out.get("kept_snps", np.nan),
                })
    return pd.DataFrame(rows)

def _resolve_df_and_order(df: Optional[pd.DataFrame], d_syn: Optional[np.ndarray], d_real: Optional[np.ndarray], sample_names: Optional[Sequence[str]], sort_by: str, top_k: Optional[int]) -> pd.DataFrame:
    """Build or validate the per-sample table, apply sort and optional top-k.
 
    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-built per-sample table. 
    d_syn : np.ndarray, optional
        Raw synthetic distance array.
    d_real : np.ndarray, optional
        Raw holdout baseline distance array. 
    sample_names : sequence of str, optional
        Sample identifiers; used only when building the table from raw
        arrays.
    sort_by : str
        Sort key. One of:
 
        - "margin": sort by ``d_real - d_syn`` descending
          (largest margin first).
        - "ratio": sort by ``d_syn / d_real`` ascending
          (smallest ratio first).
    top_k : int, optional
        If provided, truncate to the first ``top_k`` rows after sorting.
 
    Returns
    -------
    pd.DataFrame
        Sorted and optionally truncated per-sample table.
    """
    if df is None:
        if d_syn is None or d_real is None:
            raise ValueError("Provide either `df` or both `d_syn` and `d_real`.")
        df = per_sample_pli_table(d_syn, d_real, sample_names=sample_names)
    else:
        if d_syn is not None or d_real is not None:
            raise ValueError("Pass `df` OR raw arrays, not both.")
 
    sort_map = {
        "margin": ("margin_dreal_minus_dsyn", False),
        "ratio":  ("ratio_dsyn_dreal",        True),
    }
    if sort_by not in sort_map:
        raise ValueError(f"sort_by must be one of {list(sort_map)}")
 
    col, asc = sort_map[sort_by]
    df = df.sort_values(col, ascending=asc).reset_index(drop=True)
 
    if top_k is not None:
        df = df.iloc[: int(top_k)].reset_index(drop=True)
 
    return df
 
 
def plot_dsyn_dreal_per_sample(d_syn: Optional[np.ndarray] = None, d_real: Optional[np.ndarray] = None, *, df: Optional[pd.DataFrame] = None, sample_names: Optional[Sequence[str]] = None, sort_by: str = "margin", top_k: Optional[int] = None, ax: Optional[plt.Axes] = None, title: str = "PLI per-sample distances") -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot per-holdout-sample ``d_syn`` and ``d_real`` as a paired line chart.
 
    Parameters
    ----------
    d_syn : np.ndarray, optional
        Per-sample synthetic distance array. 
    d_real : np.ndarray, optional
        Per-sample holdout baseline distance array. 
    df : pd.DataFrame, optional
        Pre-built per-sample table from ``per_sample_pli_table``.
    sample_names : sequence of str, optional
        Sample identifiers; used only when building the table from raw
        arrays.
    sort_by : str, default="margin"
        How to order samples on the x-axis. One of ``"margin"``,
        ``"ratio"`` (see ``_resolve_df_and_order``).
    top_k : int, optional
        Restrict the plot to the ``top_k`` most identifiable samples
        after sorting. Useful for large holdout cohorts.
    ax : plt.Axes, optional
        Existing axes to draw on. A new figure is created if ``None``.
    title : str, default="PLI per-sample distances"
        Axes title.
 
    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    ax : plt.Axes
        The axes containing the plot.
    df_plot : pd.DataFrame
        The sorted and optionally truncated per-sample DataFrame used
        to produce the plot.
    """
    df_plot = _resolve_df_and_order(df, d_syn, d_real, sample_names, sort_by, top_k)
 
    x      = np.arange(len(df_plot))
    labels = df_plot["sample"].tolist()
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(x) * 0.25), 4))
    else:
        fig = ax.figure
 
    ax.plot(x, df_plot["d_real"], marker="o", label="$d_{real}$ (HO baseline)")
    ax.plot(x, df_plot["d_syn"],  marker="o", label="$d_{syn}$ (nearest SYN)")
    ax.set_xlabel("HO samples")
    ax.set_ylabel("Distance")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.2)
 
    return fig, ax, df_plot
 
 
def plot_identifiability_scatter(d_syn: Optional[np.ndarray] = None, d_real: Optional[np.ndarray] = None, *, df: Optional[pd.DataFrame] = None, sample_names: Optional[Sequence[str]] = None, annotate_top_k: int = 0, ax: Optional[plt.Axes] = None, title: str = "PLI identifiability") -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Scatter plot of ``d_syn`` vs ``d_real`` with the identity diagonal.
 
    Parameters
    ----------
    d_syn : np.ndarray, optional
        Per-sample synthetic distance array. 
    d_real : np.ndarray, optional
        Per-sample holdout baseline distance array. 
    df : pd.DataFrame, optional
        Pre-built per-sample table from ``per_sample_pli_table``.
    sample_names : sequence of str, optional
        Sample identifiers; used only when building the table from raw
        arrays.
    annotate_top_k : int, default=0
        If > 0, annotate the ``top_k`` samples with the smallest
        ``rank_s_ratio`` (most at risk) with their sample identifier.
    ax : plt.Axes, optional
        Existing axes to draw on. 
    title : str, default="PLI identifiability"
        Axes title prefix. The fraction of samples below the diagonal
        is appended automatically.
 
    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    ax : plt.Axes
        The axes containing the plot.
    df_ranked : pd.DataFrame
        The full per-sample DataFrame sorted by ``rank_s_ratio``
        (most at risk first).
    """
    # `_resolve_df_and_order` only exposes "margin"/"ratio" sort keys now,
    # so build/validate the table via that helper (order doesn't matter
    # here since we re-sort by risk rank immediately below).
    df_full = _resolve_df_and_order(df, d_syn, d_real, sample_names, "margin", None)
    df_full = df_full.sort_values("rank_s_ratio", ascending=True).reset_index(drop=True)
 
    n     = len(df_full)
    below = df_full["ap_flag"].sum()
    frac  = below / n
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
 
    ax.scatter(df_full["d_real"], df_full["d_syn"], s=18)
 
    lo = float(min(df_full["d_real"].min(), df_full["d_syn"].min()))
    hi = float(max(df_full["d_real"].max(), df_full["d_syn"].max()))
 
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="grey")
 
    ax.set_xlabel("$d_{real}$ (HO 2-NN baseline)")
    ax.set_ylabel("$d_{syn}$ (nearest SYN)")
    ax.legend()
    ax.grid(True, alpha=0.2)
 
    if annotate_top_k > 0:
        top = df_full.nsmallest(int(annotate_top_k), "rank_s_ratio")
        for _, row in top.iterrows():
            ax.annotate(str(row["sample"]), (row["d_real"], row["d_syn"]), fontsize=8)
 
    return fig, ax, df_full
 
def pli_diagnostics(G_tr: np.ndarray,G_ho: np.ndarray, G_syn: np.ndarray,*, ho_names: Optional[Sequence[str]] = None, n_components: int = 10, q: float = 0.01, random_seed: int = 123, boot: int = 200) -> Dict[str, object]:
    """Run a full PLI diagnostic suite in a single call.
 
    Parameters
    ----------
    G_tr : np.ndarray
        Training genotype matrix.
    G_ho : np.ndarray
        Holdout genotype matrix.
    G_syn : np.ndarray
        Synthetic genotype matrix.
    ho_names : sequence of str, optional
        Identifiers for holdout samples. If ``None``, samples are
        labelled ``"HO_0"``, ``"HO_1"``, etc.
    n_components : int, default=10
        Number of PCA components.
    q : float, default=0.01
        Quantile level for the ``r_p`` sub-score.
    random_seed : int, default=123
        Random seed.
    boot : int, default=100
        Number of bootstrap resamples for ``bootstrap_pli_consistency``.
 
    Returns
    -------
    dict
        Dictionary with the following keys:
 
        - "pli" (dict): the full output of ``compute_pli``,
          including all sub-scores, distance arrays, and metadata.
        - "driver" (str): which sub-score drove the final PLI
          value -- ``"ratio_tail"`` when ``r_p >= r_A`` (the quantile
          tail proximity signal dominated), or ``"adv_proximity"`` when
          ``r_A > r_p`` (the adversarial proximity signal dominated).
          Useful for deciding which diagnostic plot to prioritise.
        - "per_sample" (pd.DataFrame): output of
          ``per_sample_pli_table``, sorted by identifiability risk.
          Pass directly as ``df`` to ``plot_dsyn_dreal_per_sample`` or
          ``plot_identifiability_scatter``.
        - "boot" (pd.DataFrame): output of
          ``bootstrap_pli_consistency``, one row per bootstrap draw.
          Summarise with ``.describe()`` or ``.quantile([0.025, 0.975])``
          for confidence intervals on PLI sub-scores.
    """
    out = compute_pli(
        G_tr, G_ho, G_syn,
        n_components=n_components,
        q=q,
        random_seed=random_seed,
    )
 
    df_samples = per_sample_pli_table(out["d_syn"], out["d_real"], sample_names=ho_names)
    df_boot    = bootstrap_pli_consistency(out["d_syn"], out["d_real"], q=q, boot = boot, random_seed=random_seed)
 
    return {
        "pli":        out,
        "driver":     "ratio_tail" if out["r_p"] >= out["r_A"] else "adv_proximity",
        "per_sample": df_samples,
        "boot":       df_boot,
    }
 
__all__ = ["per_sample_pli_table",
           "bootstrap_pli_consistency",
           "scan_pli_stability",
           "_resolve_df_and_order",
           "plot_dsyn_dreal_per_sample",
           "plot_identifiability_scatter",
           "pli_diagnostics",]
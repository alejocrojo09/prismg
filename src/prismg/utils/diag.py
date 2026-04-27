from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prismg.metrics.pli import compute_pli
from prismg.score import clamp01


def per_sample_pli_table(d_syn: np.ndarray, d_real: np.ndarray, sample_names: Optional[Sequence[str]] = None, eps: float = 1e-12) -> pd.DataFrame:
    """
    Build per-HO-sample PLI diagnostics table.

    Columns
    -------
    sample                  : sample identifier
    d_syn                   : distance to nearest synthetic neighbour
    d_real                  : 2-NN baseline distance in HO set
    ratio_dsyn_dreal        : d_syn / d_real  (< 1 => risk)
    margin_dreal_minus_dsyn : d_real - d_syn  (> 0 => risk)
    s_ratio                 : clamp(1 - ratio, 0, 1)  — per-sample proximity score
    ap_flag                 : 1 if d_syn < d_real (adversarial-proximity candidate)
    rank_s_ratio            : rank by s_ratio descending (1 = most risky)
    rank_margin             : rank by margin descending
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


def bootstrap_pli_consistency(d_syn: np.ndarray, d_real: np.ndarray, q: float = 0.01, boot: int = 200, seed: int = 123) -> pd.DataFrame:
    """
    Bootstrap over HO samples to assess PLI stability.

    Resamples indices B times and recomputes rho_q, r_p, A, r_A, PLI
    without re-running PCA/NN.

    Returns
    -------
    DataFrame with columns: rho_q, r_p, A, r_A, PLI  (one row per bootstrap draw).
    Caller is responsible for summarising (e.g. .describe() or quantile CIs).
    """
    rng   = np.random.default_rng(seed)
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
    """
    Recompute full PLI across a grid of hyperparameters and seeds.

    Useful for assessing sensitivity to PCA dimensionality and quantile threshold.

    Returns
    -------
    DataFrame with columns: n_components, q, seed, rho_q, r_p, A, r_A, PLI, kept_snps.
    """
    rows = []
    for nc in n_components_grid:
        for q in q_grid:
            for s in seeds:
                out = compute_pli(
                    G_tr, G_ho, G_syn,
                    n_components=nc,
                    random_state=s,
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
    """
    Internal helper: build or validate the sample table, apply sort and top-k.

    Accepts either a pre-built DataFrame (from per_sample_pli_table) or raw
    d_syn / d_real arrays. Returns a sorted, optionally truncated DataFrame.
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
        "index":  (None,                       None),
        "rank":   ("rank_s_ratio",             True),
    }
    if sort_by not in sort_map:
        raise ValueError(f"sort_by must be one of {list(sort_map)}")

    col, asc = sort_map[sort_by]
    df = df.sort_values(col, ascending=asc).reset_index(drop=True) if col else df

    if top_k is not None:
        df = df.iloc[: int(top_k)].reset_index(drop=True)

    return df


def plot_dsyn_dreal_per_sample(d_syn: Optional[np.ndarray] = None, d_real: Optional[np.ndarray] = None, *, df: Optional[pd.DataFrame] = None, sample_names: Optional[Sequence[str]] = None, sort_by: str = "margin", top_k: Optional[int] = None, ax: Optional[plt.Axes] = None, title: str = "PLI per-sample distances") -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Line plot of d_syn and d_real per HO sample.

    Parameters
    ----------
    d_syn, d_real : raw distance arrays  — supply these OR `df`, not both.
    df            : pre-built per_sample_pli_table DataFrame (avoids recomputation).
    sample_names  : used only when raw arrays are supplied.
    sort_by       : "margin" (d_real−d_syn desc), "ratio" (d_syn/d_real asc),
                    "rank" (rank_s_ratio asc), "index" (original order).
    top_k         : restrict plot to the k most identifiable samples.
    ax            : existing Axes to draw on; a new figure is created if None.
    title         : axes title.

    Returns
    -------
    fig, ax, df_plot  — figure, axes, and the (sorted/truncated) DataFrame used.
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
    """
    Scatter plot of d_syn (y) vs d_real (x) with identity diagonal.

    Points **below** the diagonal (d_syn < d_real) are identifiability candidates —
    their nearest synthetic neighbour is closer than their HO baseline.

    Parameters
    ----------
    d_syn, d_real : raw distance arrays — supply these OR `df`, not both.
    df            : pre-built per_sample_pli_table DataFrame (avoids recomputation).
    sample_names  : used only when raw arrays are supplied.
    annotate_top_k: annotate the k samples with the largest margin (d_real − d_syn).
    ax            : existing Axes; a new figure is created if None.
    title         : axes title prefix (fraction below diagonal is appended).

    Returns
    -------
    fig, ax, df_ranked  — figure, axes, and the full DataFrame sorted by rank_s_ratio.
    """
    df_full = _resolve_df_and_order(df, d_syn, d_real, sample_names, "rank", None)

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

def pli_diagnostics(G_tr: np.ndarray,G_ho: np.ndarray, G_syn: np.ndarray,*, ho_names: Optional[Sequence[str]] = None, n_components: int = 10, q: float = 0.01, random_state: int = 123, boot: int = 200) -> Dict[str, object]:
    """
    One-call PLI diagnostics.

    Runs compute_pli once; all downstream tables and plots reuse the resulting
    distance arrays — no redundant computation.

    Returns
    -------
    dict with keys:
      pli         : raw compute_pli output dict
      driver      : "ratio_tail" | "adv_proximity"
      per_sample  : per_sample_pli_table DataFrame
      boot        : bootstrap_pli_consistency DataFrame (raw draws)

    Plotting
    --------
    Pass ``result["per_sample"]`` directly as the `df` argument to
    plot_dsyn_dreal_per_sample and plot_identifiability_scatter to avoid
    rebuilding the table.
    """
    out = compute_pli(
        G_tr, G_ho, G_syn,
        n_components=n_components,
        q=q,
        random_state=random_state,
    )

    df_samples = per_sample_pli_table(out["d_syn"], out["d_real"], sample_names=ho_names)
    df_boot    = bootstrap_pli_consistency(out["d_syn"], out["d_real"], q=q, boot = boot, seed=random_state)

    return {
        "pli":        out,
        "driver":     "ratio_tail" if out["r_p"] >= out["r_A"] else "adv_proximity",
        "per_sample": df_samples,
        "boot":       df_boot,
    }

"""
KRI Diagnostics — plotting and statistical breakdowns for r_replay, r_IKE, r_HAP, r_SPEC.

This module keeps KRI scoring (kri.py) clean and provides richer,
reusable diagnostics for:
  - Replay (pedigree) behavior
  - Internal Kinship Excess (IKE)
  - Micro-haplotype collision distributions and positional spikes
  - Spectral inflation (spectrum shape and leading eigenvector)

All plotting uses matplotlib; the caller can decide to show or save figures.
"""

from __future__ import annotations

import bisect
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, List, Sequence, Tuple

from prism_g.grm import standardize_by_af, compute_grm
from prism_g.kri import pedigree_replay_M, hap_collision_rate


# =============================
# Shared helpers
# =============================

def _tri_upper(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri = np.triu_indices_from(K, 1)
    return tri[0], tri[1], K[tri]

def _frac_ge(K: np.ndarray, tri: Tuple[np.ndarray, np.ndarray], t: float) -> float:
    return float(np.mean(K[tri] >= t))

def _norm_spectrum(K: np.ndarray) -> np.ndarray:
    vals = np.linalg.eigvalsh(K)
    vals = np.sort(vals)[::-1]
    return vals / (np.sum(vals) + 1e-12)

def _per_column_mode_impute(W: np.ndarray) -> np.ndarray:
    """Mode-impute NaNs per column (ties broken by first max). Cast to int after."""
    Wc = W.copy()
    for j in range(Wc.shape[1]):
        col = Wc[:, j]
        mask = ~np.isnan(col)
        if np.any(mask):
            vals, cnts = np.unique(col[mask], return_counts=True)
            rep = vals[np.argmax(cnts)]
        else:
            rep = 0.0
        col[~mask] = rep
        Wc[:, j] = col
    return Wc.astype(int)

def _chrom_index_map(var_chr: Iterable) -> dict:
    """Return {chrom: [variant_indices]} keeping original order, restrict to autosomes 1..22 if numeric."""
    from collections import defaultdict
    idx_by_chr = defaultdict(list)
    for j, c in enumerate(var_chr):
        cc = str(c)
        if cc.isdigit() and 1 <= int(cc) <= 22:
            idx_by_chr[cc].append(j)
        else:
            # keep non-numeric chromosomes too, but sort later lexically after autosomes
            idx_by_chr[cc].append(j)
    return idx_by_chr

def _iter_chr_windows(var_chr: Iterable, window_k: int, stride: int) -> Iterable[Tuple[int, List[int]]]:
    """
    Yield (start_idx_global, cols) for each sliding window of size window_k with given stride,
    iterating chromosome by chromosome preserving original variant order.
    """
    idx_by_chr = _chrom_index_map(var_chr)
    # Sort numerically for 1..22 first, then others lexicographically
    def _chr_key(x: str):
        return (0, int(x)) if x.isdigit() else (1, x)
    for c in sorted(idx_by_chr.keys(), key=_chr_key):
        idxs = idx_by_chr[c]
        if len(idxs) < window_k:
            continue
        for start in range(0, len(idxs) - window_k + 1, stride):
            cols = idxs[start:start + window_k]
            yield cols[0], cols


# =============================
# Public helpers for HAP diags
# =============================

def per_window_collision_rates(
    G: np.ndarray,
    var_chr: Iterable,
    k: int = 8,
    stride: int = 4,
    min_poly: int = 6
) -> np.ndarray:
    """
    Compute a collision rate per window:
      rate = (# identical-haplotype pairs in window) / (total_pairs)
    with the same rules as hap_collision_rate:
      - only windows with at least `min_poly` polymorphic columns (nanstd>0) are kept
      - NaNs are imputed by per-column mode (like kri.hap_collision_rate)
    Returns an array of rates (one per valid window). If no valid windows, returns empty array.
    """
    G = np.asarray(G, dtype=float)
    n, m = G.shape
    total_pairs = n * (n - 1) / 2.0
    if total_pairs <= 0 or m == 0:
        return np.array([], dtype=float)

    from collections import Counter
    rates = []
    for start_idx, cols in _iter_chr_windows(var_chr, k, stride):
        W = G[:, cols]
        # polymorphism filter
        if np.sum(np.nanstd(W, axis=0) > 0) < min_poly:
            continue
        Wi = _per_column_mode_impute(W)
        keys = ['|'.join(map(str, row.tolist())) for row in Wi]
        ctr = Counter(keys)
        collisions = 0.0
        for cnt in ctr.values():
            if cnt >= 2:
                collisions += cnt * (cnt - 1) / 2.0
        rates.append((collisions / total_pairs) if total_pairs > 0 else 0.0)

    return np.array(rates, dtype=float)

def window_positions_and_rates(
    G: np.ndarray,
    var_chr: Iterable,
    k: int = 8,
    stride: int = 4,
    min_poly: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (positions, rates) where:
      - positions: global variant start index for each valid window (x-axis for spike plots)
      - rates: window collision rates (same definition as per_window_collision_rates)
    Windows not passing the polymorphism filter are skipped.
    """
    G = np.asarray(G, dtype=float)
    n, m = G.shape
    total_pairs = n * (n - 1) / 2.0
    if total_pairs <= 0 or m == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    from collections import Counter
    pos_list: List[int] = []
    rate_list: List[float] = []

    for start_idx, cols in _iter_chr_windows(var_chr, k, stride):
        W = G[:, cols]
        if np.sum(np.nanstd(W, axis=0) > 0) < min_poly:
            continue
        Wi = _per_column_mode_impute(W)
        keys = ['|'.join(map(str, row.tolist())) for row in Wi]
        ctr = Counter(keys)
        collisions = 0.0
        for cnt in ctr.values():
            if cnt >= 2:
                collisions += cnt * (cnt - 1) / 2.0

        pos_list.append(start_idx)
        rate_list.append((collisions / total_pairs) if total_pairs > 0 else 0.0)

    return np.array(pos_list, dtype=int), np.array(rate_list, dtype=float)


# =============================
# Replay diagnostics
# =============================

def plot_replay_diagnostics(
    G_tr: np.ndarray,
    G_ho: np.ndarray,
    G_syn: np.ndarray,
    theta: float = 0.125,
    tau: float = 0.02,
    n_boot: int = 200,
    seed: int = 7
) -> None:
    X_ho, _  = standardize_by_af(G_ho, G_tr)
    X_syn, _ = standardize_by_af(G_syn, G_tr)
    K_ho  = compute_grm(X_ho)
    K_syn = compute_grm(X_syn)

    tri = np.triu_indices_from(K_ho, 1)
    print(f"HO pairs ≥ {theta}: {int(np.sum(K_ho[tri] >= theta))}")

    k_ho, k_syn = K_ho[tri], K_syn[tri]
    grid = np.linspace(0.0, 0.3, 151)

    plt.figure()
    plt.plot(grid, [np.mean(k_ho  >= t) for t in grid], label="HO tail P(K≥t)")
    plt.plot(grid, [np.mean(k_syn >= t) for t in grid], label="S tail P(K≥t)")
    plt.axvline(theta, ls="--"); plt.legend(); plt.title("Kinship tails")

    close_vals = k_ho[k_ho >= theta]
    gaps = [np.min(np.abs(k_syn - v)) for v in close_vals] if close_vals.size else []
    plt.figure()
    plt.hist(gaps, bins=40, density=True); plt.axvline(tau)
    plt.title("Replay match-gaps Δ to HO-close pairs")

    def _replay_rate(Kho: np.ndarray, Ksyn: np.ndarray, th: float, tw: float) -> float:
        tri = np.triu_indices_from(Kho, 1)
        k_ho, k_syn = Kho[tri], Ksyn[tri]
        targets = k_ho[k_ho >= th]
        if targets.size == 0:
            return 0.0
        ks = np.sort(k_syn)
        hits = sum(bisect.bisect_right(ks, v + tw) > bisect.bisect_left(ks, v - tw) for v in targets)
        return hits / targets.size

    M = _replay_rate(K_ho, K_syn, theta, tau)
    rng = np.random.default_rng(seed)
    M0s = []
    for _ in range(max(200, n_boot)):
        idx = rng.permutation(K_syn.shape[0])
        Kp  = K_syn[np.ix_(idx, idx)]
        M0s.append(_replay_rate(K_ho, Kp, theta, tau))

    plt.figure()
    plt.hist(M0s, bins=30, alpha=0.7)
    plt.axvline(M)
    plt.title(f"r_replay baseline M0 vs observed M = {M:.3f}")
    p = (np.sum(np.array(M0s) >= M) + 1) / (len(M0s) + 1)
    print("Replay perm p-value:", p)


# =============================
# IKE diagnostics
# =============================

def plot_ike_diagnostics(
    G_tr: np.ndarray,
    G_ho: np.ndarray,
    G_syn: np.ndarray,
    thetas: Sequence[float] = (0.10, 0.125, 0.25),
    n_boot: int = 200,
    seed: int = 1
) -> None:
    X_ho, _  = standardize_by_af(G_ho, G_tr)
    X_syn, _ = standardize_by_af(G_syn, G_tr)
    K_ho  = compute_grm(X_ho)
    K_syn = compute_grm(X_syn)

    tri = np.triu_indices_from(K_ho, 1)
    fS = {th: _frac_ge(K_syn, tri, th) for th in thetas}

    rng = np.random.default_rng(seed)
    f0 = {th: [] for th in thetas}
    B = max(200, n_boot)
    for _ in range(B):
        idx = rng.choice(K_ho.shape[0], size=K_syn.shape[0], replace=True)
        Kb  = K_ho[np.ix_(idx, idx)]
        for th in thetas:
            f0[th].append(_frac_ge(Kb, tri, th))

    for th in thetas:
        v = np.array(f0[th], dtype=float)
        ci = (np.quantile(v, 0.025), np.quantile(v, 0.975))
        print(f"θ={th}: fS={fS[th]:.4f}, f0_mean={v.mean():.4f}, 95%CI={ci}")

    grid = np.linspace(0.0, 0.3, 151)
    tails0 = []
    for _ in range(200):
        idx = rng.choice(K_ho.shape[0], K_syn.shape[0], replace=True)
        Kb  = K_ho[np.ix_(idx, idx)]
        kb  = Kb[tri]
        tails0.append([np.mean(kb >= t) for t in grid])
    tails0 = np.array(tails0)

    plt.figure()
    plt.plot(grid, [np.mean(K_syn[tri] >= t) for t in grid], label="S")
    plt.plot(grid, np.median(tails0, 0), label="baseline (median)")
    plt.fill_between(grid, np.quantile(tails0, 0.05, 0), np.quantile(tails0, 0.95, 0), alpha=0.2, label="90% band")
    plt.legend(); plt.title("Internal kinship tail")


# =============================
# Haplotype diagnostics
# =============================

def plot_hap_diagnostics(
    G_ho: np.ndarray,
    G_syn: np.ndarray,
    var_chr: Iterable,
    k: int = 8,
    stride: int = 4,
    min_poly: int = 6,
    n_boot: int = 200,
    seed: int = 2
) -> None:
    rng = np.random.default_rng(seed)
    rates_S = per_window_collision_rates(G_syn, var_chr, k=k, stride=stride, min_poly=min_poly)

    rates0_list = []
    B = max(200, n_boot)
    for _ in range(B):
        idx = rng.choice(G_ho.shape[0], G_syn.shape[0], replace=True)
        rates0_list.append(per_window_collision_rates(G_ho[idx], var_chr, k=k, stride=stride, min_poly=min_poly))
    rates0 = np.concatenate([r for r in rates0_list if r.size])

    plt.figure()
    if rates_S.size:
        plt.hist(rates_S, bins=30, alpha=0.6, label="S")
    if rates0.size:
        plt.hist(rates0, bins=30, alpha=0.6, label="baseline")
    plt.legend(); plt.title("Window collision rate distribution")

    pos, rate = window_positions_and_rates(G_syn, var_chr, k=k, stride=stride, min_poly=min_poly)
    plt.figure()
    if rate.size:
        plt.plot(pos, rate, lw=1)
    plt.title("Collision rate vs genomic position")
    plt.xlabel("window start index"); plt.ylabel("rate")


# =============================
# Spectral diagnostics
# =============================

def plot_spectral_diagnostics(
    G_tr: np.ndarray,
    G_ho: np.ndarray,
    G_syn: np.ndarray,
    kmax: int = 20,
    n_boot: int = 200,
    seed: int = 3
) -> None:
    X_ho, _  = standardize_by_af(G_ho, G_tr)
    X_syn, _ = standardize_by_af(G_syn, G_tr)
    K_ho  = compute_grm(X_ho)
    K_syn = compute_grm(X_syn)

    s_syn = _norm_spectrum(K_syn)
    rng = np.random.default_rng(seed)
    S0 = []
    B = max(200, n_boot)
    for _ in range(B):
        idx = rng.choice(K_ho.shape[0], K_syn.shape[0], replace=True)
        S0.append(_norm_spectrum(K_ho[np.ix_(idx, idx)]))
    S0 = np.array(S0)

    plt.figure()
    plt.plot(range(1, kmax + 1), s_syn[:kmax], label="S")
    plt.plot(range(1, kmax + 1), np.median(S0, 0)[:kmax], label="baseline (median)")
    plt.fill_between(range(1, kmax + 1),
                     np.quantile(S0, 0.05, 0)[:kmax],
                     np.quantile(S0, 0.95, 0)[:kmax],
                     alpha=0.2, label="90% band")
    plt.legend(); plt.title("Normalized spectrum (top 20)")

    vals, vecs = np.linalg.eigh(K_syn)
    v1 = vecs[:, -1]
    plt.figure(); plt.hist(v1, bins=30)
    plt.title("Leading eigenvector entries (S)")


# =============================
# Exports
# =============================

__all__ = [
    "per_window_collision_rates",
    "window_positions_and_rates",
    "plot_replay_diagnostics",
    "plot_ike_diagnostics",
    "plot_hap_diagnostics",
    "plot_spectral_diagnostics",
]

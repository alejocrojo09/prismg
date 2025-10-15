"""
KRI — Kinship Replay Index 

Implements subcomponents:
- r_replay : pedigree replay
- r_IKE    : internal kinship excess
- r_HAP    : micro-haplotype collision
- r_SPEC   : spectral inflation / clustering anomalies

Final score: KRI = max(r_replay, r_IKE, r_HAP, r_SPEC)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Iterable, Optional, Tuple

from prism_g.grm import standardize_by_af, compute_grm

# =============================
# Utils
# =============================

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _symmetrize(K: np.ndarray) -> np.ndarray:
    return 0.5 * (K + K.T)


def count_close_pairs(K: np.ndarray, theta: float) -> int:
    iu = np.triu_indices_from(K, 1)
    return int(np.sum(K[iu] >= theta))


def permute_matrix_rows_cols(K: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = rng.permutation(K.shape[0])
    return K[np.ix_(idx, idx)]

# ----------------------------
# Replay (pedigree) component
# ----------------------------

def pedigree_replay_M(K_real: np.ndarray, K_syn: np.ndarray, theta: float = 0.125, tau: float = 0.02) -> float:
    iu = np.triu_indices_from(K_real, 1)
    real_vals = K_real[iu]
    syn_vals  = K_syn[iu]
    rclose = real_vals[real_vals >= theta]
    if rclose.size == 0:
        return 0.0
    syn_sorted = np.sort(syn_vals)
    from bisect import bisect_left, bisect_right
    hits = 0
    for v in rclose:
        lo, hi = v - tau, v + tau
        L, R = bisect_left(syn_sorted, lo), bisect_right(syn_sorted, hi)
        if R > L:
            hits += 1
    return float(hits) / float(rclose.size)

# ----------------------------
# Internal Kinship Excess (IKE)
# ----------------------------
def internal_kinship_excess(
    K_syn: np.ndarray,
    K_safe: Optional[np.ndarray] = None,
    K_ho: Optional[np.ndarray] = None,
    theta_list: Iterable[float] = (0.10, 0.125, 0.25),
    n_boot: int = 200,
    seed: int = 42
) -> Tuple[float, Dict[float, float]]:
    rng = np.random.default_rng(seed)
    iu = np.triu_indices_from(K_syn, 1)
    v_syn = K_syn[iu]

    def frac_ge(v: np.ndarray, t: float) -> float:
        return float(np.mean(v >= t))

    r_list = []
    breakdown: Dict[float, float] = {}
    for theta in theta_list:
        f = frac_ge(v_syn, theta)
        if K_safe is not None:
            iu_s = np.triu_indices_from(K_safe, 1)
            v0_full = K_safe[iu_s]; m = v_syn.size
            f0 = float(np.mean([frac_ge(v0_full[rng.integers(0, v0_full.size, m)], theta)
                                for _ in range(n_boot)]))
        elif K_ho is not None:
            iu_r = np.triu_indices_from(K_ho, 1)
            v0_full = K_ho[iu_r]; m = v_syn.size
            f0 = float(np.mean([frac_ge(v0_full[rng.integers(0, v0_full.size, m)], theta)
                                for _ in range(n_boot)]))
        else:
            f0 = f
        r = 0.0 if f <= f0 else (f - f0) / (1.0 - f0 + 1e-12)
        r = clamp01(r)
        r_list.append(r)
        breakdown[theta] = r
    return max(r_list) if r_list else 0.0, breakdown

# ----------------------------
# Micro-haplotype collision
# ----------------------------
def hap_collision_rate(G: np.ndarray, var_chr: Iterable, window_k: int = 8, stride: int = 4, min_poly: int = 6) -> float:
    G = np.asarray(G, dtype=float)
    n, m = G.shape
    total_pairs = n * (n - 1) / 2.0
    if total_pairs == 0 or m == 0:
        return 0.0
    from collections import defaultdict, Counter
    idx_by_chr = defaultdict(list)
    for j, c in enumerate(var_chr):
        cc = str(c)
        if cc.isdigit() and 1 <= int(cc) <= 22:
            idx_by_chr[cc].append(j)
    collisions, n_windows = 0.0, 0
    for c in sorted(idx_by_chr, key=lambda x: int(x)):
        idxs = idx_by_chr[c]
        if len(idxs) < window_k:
            continue
        for start in range(0, len(idxs) - window_k + 1, stride):
            cols = idxs[start:start + window_k]
            W = G[:, cols]
            if np.sum(np.nanstd(W, axis=0) > 0) < min_poly:
                continue
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
            Wc = Wc.astype(int)
            keys = ['|'.join(map(str, row.tolist())) for row in Wc]
            ctr = Counter(keys)
            for cnt in ctr.values():
                if cnt >= 2:
                    collisions += cnt * (cnt - 1) / 2.0
            n_windows += 1
    if n_windows == 0:
        return 0.0
    return float((collisions / n_windows) / total_pairs)


def hap_collision_risk(
    G_syn: np.ndarray,
    var_chr: Iterable,
    G_safe: Optional[np.ndarray] = None,
    G_ho: Optional[np.ndarray] = None,
    window_k: int = 8, stride: int = 4, min_poly: int = 6,
    n_boot: int = 200, seed: int = 1
) -> Tuple[float, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    c = hap_collision_rate(G_syn, var_chr, window_k, stride, min_poly)
    def boot_rate(Gb: np.ndarray) -> float:
        return hap_collision_rate(Gb, var_chr, window_k, stride, min_poly)
    if G_safe is not None:
        c0s = [boot_rate(G_safe) for _ in range(n_boot)]
    elif G_ho is not None:
        c0s = [boot_rate(G_ho) for _ in range(n_boot)]
    else:
        c0s = [c] * n_boot
    c0 = float(np.mean(c0s))
    r = 0.0 if c <= c0 else (c - c0) / (c + 1e-12)
    return clamp01(r), dict(obs=c, base=c0)

# ----------------------------
# Spectral inflation
# ----------------------------
def spectral_inflation(K: np.ndarray, iters: int = 50, seed: int = 0) -> float:
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    if n == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    v = rng.normal(size=n)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(iters):
        v = K @ v
        nv = np.linalg.norm(v)
        if nv == 0.0:
            break
        v /= nv
    lam1 = float(v @ (K @ v) / (v @ v + 1e-12))
    tr = float(np.trace(K))
    return lam1 / (tr + 1e-12)


def spectral_risk(K_syn: np.ndarray, K_base: Optional[np.ndarray] = None,
                  n_boot: int = 200, seed: int = 7) -> Tuple[float, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    K_syn = _symmetrize(np.asarray(K_syn, dtype=float))
    s = spectral_inflation(K_syn)
    if K_base is None:
        s0 = s
    else:
        K_base = _symmetrize(np.asarray(K_base, dtype=float))
        n_syn, n_base = K_syn.shape[0], K_base.shape[0]
        replace = n_syn > n_base
        s0s = []
        for _ in range(n_boot):
            idx = rng.choice(n_base, size=n_syn, replace=replace)
            Kb = K_base[np.ix_(idx, idx)]
            s0s.append(spectral_inflation(Kb))
        s0 = float(np.mean(s0s))
    r = 0.0 if s <= s0 else (s - s0) / (s + 1e-12)
    return clamp01(r), dict(obs=s, base=s0)

# ----------------------------
# Top-level KRI
# ----------------------------
def compute_kri(G_tr, G_ho, G_syn, var_chr,
                G_safe=None, theta=0.125, tau=0.02,
                ike_thetas=(0.10, 0.125, 0.25),
                n_boot=200, seed=42, use_replay=True) -> Dict:
    rng = np.random.default_rng(seed)
    X_tr, _  = standardize_by_af(G_tr, G_tr)
    X_ho, _  = standardize_by_af(G_ho, G_tr)
    X_syn, _ = standardize_by_af(G_syn, G_tr)
    K_ho  = compute_grm(X_ho)
    K_syn = compute_grm(X_syn)
    K_safe = None if G_safe is None else compute_grm(standardize_by_af(G_safe, G_tr)[0])
    if use_replay and count_close_pairs(K_ho, theta) > 0:
        M  = pedigree_replay_M(K_ho, K_syn, theta=theta, tau=tau)
        M0 = np.mean([pedigree_replay_M(K_ho, permute_matrix_rows_cols(K_syn, rng), theta, tau)
                      for _ in range(max(50, n_boot//2))])
        r_replay = clamp01(0.0 if M <= M0 else (M - M0) / (1.0 - M0 + 1e-12))
    else:
        r_replay = 0.0
    r_IKE, ike_breakdown = internal_kinship_excess(K_syn, K_safe=K_safe, K_ho=K_ho,
                                                  theta_list=ike_thetas, n_boot=n_boot, seed=seed+1)
    r_HAP, hap_info = hap_collision_risk(G_syn, var_chr, G_safe=G_safe, G_ho=G_ho,
                                         window_k=8, stride=4, min_poly=6,
                                         n_boot=max(100, n_boot//2), seed=seed+2)
    K_base_for_spec = K_safe if K_safe is not None else K_ho
    r_SPEC, spec_info = spectral_risk(K_syn, K_base_for_spec,
                                      n_boot=max(100, n_boot//2), seed=seed+3)
    KRI = max(r_replay, r_IKE, r_HAP, r_SPEC)
    return {"r_replay": float(r_replay), "r_IKE": float(r_IKE),
            "r_HAP": float(r_HAP), "r_SPEC": float(r_SPEC),
            "KRI": float(KRI),
            "debug": {"ike_breakdown": ike_breakdown,
                      "hap": hap_info, "spec": spec_info}}


__all__ = [
    "compute_kri",
    "pedigree_replay_M",
    "internal_kinship_excess",
    "hap_collision_rate",
    "hap_collision_risk",
    "spectral_inflation",
    "spectral_risk",
]
from __future__ import annotations

import numpy as np
from typing import Dict, Iterable, Tuple

from prismg.utils.grm import standardize_by_af, compute_grm, _symmetrize
from prismg.score import clamp01

def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen-Shannon divergence between two discrete probability vectors.
    Returns a value in [0, log(2)] using natural logs.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log((a[mask] + eps) / (b[mask] + eps)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def replay_M_js(K_real: np.ndarray, K_syn: np.ndarray, theta: float = 0.125, n_bins: int = 25, upper: float = 0.5) -> float:
    """
    Replay score based on similarity between the close-kin tails of the real and
    synthetic kinship spectra.

    We extract pairwise kinship values >= theta from both matrices, build normalized
    histograms over [theta, upper], and define replay similarity as:

        M = 1 - JS(P_real_close, P_syn_close) / log(2)

    so that:
        M = 1   means identical close-kin tail distributions
        M = 0   means maximally different close-kin tail distributions
    """
    iu_r = np.triu_indices_from(K_real, 1)
    iu_s = np.triu_indices_from(K_syn, 1)

    real_vals = np.asarray(K_real[iu_r], dtype=float)
    syn_vals  = np.asarray(K_syn[iu_s], dtype=float)

    rclose = real_vals[real_vals >= theta]
    sclose = syn_vals[syn_vals >= theta]

    if rclose.size == 0 or sclose.size == 0:
        return 0.0

    # Clip to a biologically relevant range
    rclose = np.clip(rclose, theta, upper)
    sclose = np.clip(sclose, theta, upper)

    bins = np.linspace(theta, upper, n_bins + 1)

    p_real, _ = np.histogram(rclose, bins=bins, density=False)
    p_syn, _  = np.histogram(sclose, bins=bins, density=False)

    js = _js_divergence(p_real.astype(float), p_syn.astype(float))
    sim = 1.0 - (js / np.log(2.0))  # normalize to [0,1]

    return float(clamp01(sim))


def replay_risk(K_ho: np.ndarray, K_syn: np.ndarray, theta: float = 0.125, n_bins: int = 25, upper: float = 0.5, n_boot: int = 200, seed: int = 123, G_ho: np.ndarray | None = None, G_tr: np.ndarray | None = None,) -> Tuple[float, Dict[str, float]]:
    """
    Replay risk based on close-kin tail similarity between real and synthetic
    kinship spectra.

    Null hypothesis
    ---------------
    The synthetic close-kin tail resembles the real close-kin tail no more
    than expected under bootstrap resampling of the HO population.

    Bootstrap null (preferred — pass G_ho and G_tr)
    ------------------------------------------------
    Resample individuals from G_ho with replacement, re-standardise each
    draw using training AF from G_tr, recompute the GRM, and measure how
    similar the resampled HO tail is to the SYN tail. This corresponds to
    the correct generative null: what M would we observe if SYN were just
    a random draw from the real population?

    Fallback (G_ho or G_tr not provided)
    -------------------------------------
    Resample pairwise kinship values from K_syn (original behaviour).
    Less statistically principled but does not require raw genotypes.

    Risk score
    ----------
    r = max(0, (M_obs - M0) / (1 - M0))
    """
    rng  = np.random.default_rng(seed)
    M    = replay_M_js(K_ho, K_syn, theta=theta, n_bins=n_bins, upper=upper)
    iu_r = np.triu_indices_from(K_ho,  1)
    iu_s = np.triu_indices_from(K_syn, 1)
    n_real_close = int(np.sum(np.asarray(K_ho[iu_r], dtype=float) >= theta))
    n_syn_close  = int(np.sum(np.asarray(K_syn[iu_s], dtype=float) >= theta))

    if n_boot == 0:
        return float(clamp01(M)), {
            "obs": float(M), "base": float(M), "sd": 0.0,
            "n_real_close": n_real_close, "n_syn_close": n_syn_close,
            "n_bins": int(n_bins), "upper": float(upper),
        }

    if G_ho is not None and G_tr is not None:
        # Genotype-level bootstrap null (correct)
        n_ho = G_ho.shape[0]
        M0_samples = []
        for _ in range(n_boot):
            idx    = rng.integers(0, n_ho, n_ho)
            Xb, _  = standardize_by_af(G_ho[idx], G_tr)
            Kb     = compute_grm(Xb)
            M0_samples.append(replay_M_js(Kb, K_syn,
                                          theta=theta, n_bins=n_bins, upper=upper))
    else:
        # Fallback: resample SYN pairwise kinship values
        syn_vals = np.asarray(K_syn[iu_s], dtype=float)
        if syn_vals.size == 0:
            return 0.0, dict(obs=float(M), base=0.0, sd=0.0,
                             n_real_close=n_real_close, n_syn_close=n_syn_close,
                             n_bins=int(n_bins), upper=float(upper))

        def _M_from_synvals(sv: np.ndarray) -> float:
            real_vals = np.asarray(K_ho[iu_r], dtype=float)
            rc = real_vals[real_vals >= theta]
            sc = sv[sv >= theta]
            if rc.size == 0 or sc.size == 0:
                return 0.0
            bins = np.linspace(theta, upper, n_bins + 1)
            p_r, _ = np.histogram(np.clip(rc, theta, upper), bins=bins)
            p_s, _ = np.histogram(np.clip(sc, theta, upper), bins=bins)
            js = _js_divergence(p_r.astype(float), p_s.astype(float))
            return float(clamp01(1.0 - js / np.log(2.0)))

        M0_samples = [
            _M_from_synvals(rng.choice(syn_vals, size=syn_vals.size, replace=True))
            for _ in range(n_boot)
        ]

    M0_samples = np.array(M0_samples, dtype=float)
    M0  = float(np.mean(M0_samples))
    sd0 = float(np.std(M0_samples, ddof=1)) if n_boot > 1 else 0.0
    r   = 0.0 if M <= M0 else (M - M0) / (1.0 - M0 + 1e-12)

    return float(clamp01(r)), {
        "obs": float(M), "base": float(M0), "sd": float(sd0),
        "n_real_close": n_real_close, "n_syn_close": n_syn_close,
        "n_bins": int(n_bins), "upper": float(upper),
    }


# -------------------------
# IKE: tail mass vs holdout
# -------------------------

def internal_kinship_excess(K_syn: np.ndarray, K_ho: np.ndarray, theta_list: Iterable[float] = (0.10, 0.125, 0.25), n_boot: int = 200, seed: int = 123, G_ho: np.ndarray | None = None, G_tr: np.ndarray | None = None,) -> Tuple[float, Dict[float, float]]:
    """
    Internal Kinship Excess: for each theta, compare fraction of SYN pairs
    >= theta against a bootstrap baseline from the HO kinship spectrum.

    Bootstrap null (preferred — pass G_ho and G_tr)
    ------------------------------------------------
    Resample individuals from G_ho, re-standardise with training AF,
    recompute GRM, measure close-kin fraction. Shares bootstrap draws
    across all thetas for efficiency.

    Fallback (G_ho or G_tr not provided)
    -------------------------------------
    Resample pairwise kinship values from K_ho (original behaviour).
    """
    rng    = np.random.default_rng(seed)
    iu_syn = np.triu_indices_from(K_syn, 1)
    iu_ho  = np.triu_indices_from(K_ho,  1)
    v_syn    = np.asarray(K_syn[iu_syn], dtype=float)
    v0_full  = np.asarray(K_ho[iu_ho],  dtype=float)

    if v_syn.size == 0 or v0_full.size == 0:
        return 0.0, {float(t): 0.0 for t in theta_list}

    def frac_ge(v: np.ndarray, t: float) -> float:
        return float(np.mean(v >= t))

    # Pre-compute bootstrap baseline fracs once (shared across thetas)
    if G_ho is not None and G_tr is not None and n_boot > 0:
        n_ho_ind = G_ho.shape[0]
        boot_fracs: Dict[float, list] = {float(t): [] for t in theta_list}
        for _ in range(n_boot):
            idx   = rng.integers(0, n_ho_ind, n_ho_ind)
            Xb, _ = standardize_by_af(G_ho[idx], G_tr)
            Kb    = compute_grm(Xb)
            iu_b  = np.triu_indices_from(Kb, 1)
            vb    = np.asarray(Kb[iu_b], dtype=float)
            for t in theta_list:
                boot_fracs[float(t)].append(frac_ge(vb, t))
        use_genotype_boot = True
    else:
        use_genotype_boot = False

    r_list   = []
    breakdown: Dict[float, float] = {}

    for theta in theta_list:
        f = frac_ge(v_syn, theta)

        if use_genotype_boot:
            f0 = float(np.mean(boot_fracs[float(theta)]))
        else:
            # Fallback: resample HO kinship values
            f0 = float(np.mean([
                frac_ge(v0_full[rng.integers(0, v0_full.size, v_syn.size)], theta)
                for _ in range(n_boot)
            ]))

        r = 0.0 if f <= f0 else (f - f0) / (1.0 - f0 + 1e-12)
        r = float(clamp01(r))
        r_list.append(r)
        breakdown[float(theta)] = r

    return (max(r_list) if r_list else 0.0), breakdown


# -------------------------
# HAP: micro-haplotype reuse
# -------------------------

def hap_collision_rate(G: np.ndarray, var_chr: Iterable, *, window_k: int = 8, stride: int = 4, min_poly: int = 6) -> Tuple[float, float]:
    """
    Computes average and max window collision rate across autosomes.
    """
    G = np.asarray(G, dtype=float)
    n, m = G.shape
    total_pairs = n * (n - 1) / 2.0
    if total_pairs == 0 or m == 0:
        return 0.0, 0.0

    from collections import defaultdict, Counter
    idx_by_chr = defaultdict(list)

    for j, c in enumerate(var_chr):
        cc = str(c)
        if cc.isdigit() and 1 <= int(cc) <= 22:
            idx_by_chr[cc].append(j)

    rho_list: list[float] = []

    for cchr in sorted(idx_by_chr, key=lambda x: int(x)):
        idxs = idx_by_chr[cchr]
        if len(idxs) < window_k:
            continue

        for start in range(0, len(idxs) - window_k + 1, stride):
            cols = idxs[start:start + window_k]
            W = G[:, cols]

            if np.sum(np.nanstd(W, axis=0) > 0) < min_poly:
                continue

            Wc = W.copy()
            for jj in range(Wc.shape[1]):
                col = Wc[:, jj]
                mask = ~np.isnan(col)
                if np.any(mask):
                    vals, cnts = np.unique(col[mask], return_counts=True)
                    rep = vals[np.argmax(cnts)]
                else:
                    rep = 0.0
                col[~mask] = rep
                Wc[:, jj] = col
            Wc = Wc.astype(int)

            keys = ['|'.join(map(str, row.tolist())) for row in Wc]
            ctr = Counter(keys)

            Cw = 0.0
            for cnt in ctr.values():
                if cnt >= 2:
                    Cw += cnt * (cnt - 1) / 2.0

            rho_list.append(float(Cw / total_pairs))

    if not rho_list:
        return 0.0, 0.0

    return float(np.mean(rho_list)), float(np.max(rho_list))


def hap_collision_risk(G_syn: np.ndarray, var_chr: Iterable, G_ho: np.ndarray, *, window_k: int = 8, stride: int = 4, min_poly: int = 6, n_boot: int = 200, seed: int = 123) -> Tuple[float, Dict[str, float]]:
    """
    Haplotype collision risk:
    Compare synthetic collision rate against a bootstrap baseline from holdout genomes
    (resampling individuals with replacement).
    """
    rng = np.random.default_rng(seed)

    c, cmax = hap_collision_rate(G_syn, var_chr, window_k=window_k, stride=stride, min_poly=min_poly)

    n_ho = G_ho.shape[0]
    if n_ho <= 1:
        return 0.0, dict(obs=float(c), base=0.0, cmax=float(cmax))

    c0s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_ho, n_ho)
        Gb = np.asarray(G_ho[idx, :], dtype=float)
        cb, _ = hap_collision_rate(Gb, var_chr, window_k=window_k, stride=stride, min_poly=min_poly)
        c0s.append(cb)

    c0 = float(np.mean(c0s))

    eps = 1e-12
    if c <= c0 or cmax <= c0 + eps:
        r = 0.0
    else:
        r = (c - c0) / (cmax - c0 + eps)

    return float(clamp01(r)), dict(obs=float(c), base=float(c0), cmax=float(cmax))


# -------------------------
# SPEC: spectral inflation
# -------------------------

def spectral_inflation(K: np.ndarray) -> float:
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    if n == 0:
        return 0.0

    w = np.linalg.eigvalsh(K)
    lam1 = float(w[-1])
    tr = float(np.trace(K))
    return lam1 / (tr + 1e-12)

def spectral_risk(K_syn: np.ndarray, K_base: np.ndarray) -> Tuple[float, Dict[str, float]]:
    K_syn = _symmetrize(np.asarray(K_syn, dtype=float))
    s = spectral_inflation(K_syn)

    K_base = _symmetrize(np.asarray(K_base, dtype=float))
    s0 = spectral_inflation(K_base)
    r = 0.0 if s <= s0 else (s - s0) / (s + 1e-12)
    return clamp01(r), dict(obs=s, base=s0)


# -------------------------
# KRI wrapper
# -------------------------

def compute_kri(G_tr, G_ho, G_syn, var_chr, theta: float = 0.125, replay_n_bins: int = 25, replay_upper: float = 0.5, ike_thetas: Iterable[float] = (0.10, 0.125, 0.25), n_boot: int = 200, seed: int = 123) -> Dict:
    # standardize by training AF
    X_tr, _  = standardize_by_af(G_tr, G_tr)
    X_ho, _  = standardize_by_af(G_ho, G_tr)
    X_syn, _ = standardize_by_af(G_syn, G_tr)

    K_ho  = compute_grm(X_ho)
    K_syn = compute_grm(X_syn)

    # Replay: close-kin tail distribution similarity
    # G_ho and G_tr passed for genotype-level bootstrap null
    r_replay, replay_info = replay_risk(
        K_ho=K_ho, K_syn=K_syn,
        theta=theta, n_bins=replay_n_bins, upper=replay_upper,
        n_boot=n_boot, seed=seed,
        G_ho=G_ho, G_tr=G_tr,
    )

    # Internal kinship excess
    # G_ho and G_tr passed for genotype-level bootstrap null
    r_IKE, ike_breakdown = internal_kinship_excess(
        K_syn, K_ho=K_ho,
        theta_list=ike_thetas, n_boot=n_boot, seed=seed,
        G_ho=G_ho, G_tr=G_tr,
    )

    # Micro-haplotype collisions
    r_HAP, hap_info = hap_collision_risk(G_syn, var_chr, G_ho=G_ho, window_k=8, stride=4, min_poly=6, n_boot=n_boot, seed=seed)

    # Spectral anomaly
    r_SPEC, spec_info = spectral_risk(K_syn, K_ho)

    # Aggregate KRI
    KRI = max(r_replay, r_IKE, r_HAP, r_SPEC)

    return {
        "r_replay": float(r_replay),
        "r_IKE":    float(r_IKE),
        "r_HAP":    float(r_HAP),
        "r_SPEC":   float(r_SPEC),
        "KRI":      float(KRI),
        "debug": {
            "replay":        replay_info,
            "ike_breakdown": ike_breakdown,
            "hap":           hap_info,
            "spec":          spec_info,
            "params": {
                "theta":         float(theta),
                "replay_n_bins": int(replay_n_bins),
                "replay_upper":  float(replay_upper),
                "ike_thetas":    tuple(float(t) for t in ike_thetas),
                "n_boot":        int(n_boot),
                "seed":          int(seed),
            },
        },
    }


__all__ = [
    "compute_kri",
    "replay_M_js",
    "replay_risk",
    "internal_kinship_excess",
    "hap_collision_rate",
    "hap_collision_risk",
    "spectral_inflation",
    "spectral_risk",
]

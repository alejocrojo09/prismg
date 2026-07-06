from __future__ import annotations

import numpy as np
from typing import Dict, Iterable, Tuple

from prismg.utils.grm import standardize_by_af, compute_grm, _symmetrize
from score import clamp01, _aggregate

def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Compute the Jensen-Shannon divergence between two non-negative vectors.
 
    Parameters
    ----------
    p, q : np.ndarray
        Non-negative count or probability vectors of the same length.
    eps : float, default=1e-12
        Small constant added to denominators and log arguments to avoid
        division by zero or ``log(0)``.
 
    Returns
    -------
    float
        JS divergence in [0, log(2)]. 
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
    """Measure how closely the synthetic close-kin tail matches the real one.
 
    Parameters
    ----------
    K_real : np.ndarray
        Square symmetric GRM for the real (holdout) cohort.
    K_syn : np.ndarray
        Square symmetric GRM for the synthetic cohort.
    theta : float, default=0.125
        Kinship threshold above which a pair is considered "close kin"
        in the GRM scale used. Only pairs at or above this value enter
        the histogram comparison.
    n_bins : int, default=25
        Number of histogram bins over [``theta``, ``upper``].
    upper : float, default=0.5
        Upper clip value for the close-kin distribution, defining the
        biologically relevant range.

    Returns
    -------
    float
        Close-kin distribution similarity in [0, 1].
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

def replay_risk(G_ho: np.ndarray, G_syn: np.ndarray, G_tr: np.ndarray, theta: float = 0.125, n_bins: int = 25, upper: float = 0.5, n_boot: int = 100, random_seed: int = 123,) -> Tuple[float, Dict[str, float]]:
    """Compute a bootstrap-calibrated Replay risk score for a synthetic cohort.
 
    Parameters
    ----------
    G_ho : np.ndarray
        Holdout genotype matrix.
    G_syn : np.ndarray
        Synthetic genotype matrix to evaluate.
    G_tr : np.ndarray
        Training genotype matrix. Used only for
        AF standardisation.
    theta : float, default=0.125
        Close-kin threshold in the GRM scale.
    n_bins : int, default=25
        Number of histogram bins.
    upper : float, default=0.5
        Upper clip for the close-kin range.
    n_boot : int, default=100
        Number of bootstrap replicates for the null.
    random_seed : int, default=123
        Random seed.
 
    Returns
    -------
    risk : float
        Calibrated Replay risk score in [0, 1].
    info : dict
        Diagnostic dictionary with keys:
 
        - "obs": observed ``M`` (raw JS similarity score).
        - "base": bootstrap null mean ``M0``.
        - "sd": bootstrap null standard deviation.
        - "n_real_close": number of holdout pairs above ``theta``.
        - "n_syn_close": number of synthetic pairs above ``theta``.
        - "n_bins": histogram bin count used.
        - "upper": upper clip value used.
    """
    rng = np.random.default_rng(random_seed)
 
    # Compute GRMs from genotypes standardised by training AFs
    X_ho,  _ = standardize_by_af(G_ho,  G_tr)
    X_syn, _ = standardize_by_af(G_syn, G_tr)
    K_ho  = compute_grm(X_ho)
    K_syn = compute_grm(X_syn)
 
    M    = replay_M_js(K_ho, K_syn, theta=theta, n_bins=n_bins, upper=upper)
    iu_r = np.triu_indices_from(K_ho,  1)
    iu_s = np.triu_indices_from(K_syn, 1)
    n_real_close = int(np.sum(np.asarray(K_ho[iu_r],  dtype=float) >= theta))
    n_syn_close  = int(np.sum(np.asarray(K_syn[iu_s], dtype=float) >= theta))
 
    if n_boot == 0:
        return float(clamp01(M)), {
            "obs": float(M), "base": float(M), "sd": 0.0,
            "n_real_close": n_real_close, "n_syn_close": n_syn_close,
            "n_bins": int(n_bins), "upper": float(upper),
        }
 
    # Genotype-level bootstrap null: resample holdout individuals,
    # recompute GRM from scratch each replicate
    n_ho       = G_ho.shape[0]
    M0_samples = []
    for _ in range(n_boot):
        idx   = rng.integers(0, n_ho, n_ho)
        Xb, _ = standardize_by_af(G_ho[idx], G_tr)
        Kb    = compute_grm(Xb)
        M0_samples.append(
            replay_M_js(Kb, K_syn, theta=theta, n_bins=n_bins, upper=upper)
        )
 
    M0_samples = np.array(M0_samples, dtype=float)
    M0  = float(np.mean(M0_samples))
    sd0 = float(np.std(M0_samples, ddof=1)) if n_boot > 1 else 0.0
    r   = 0.0 if M <= M0 else (M - M0) / (1.0 - M0 + 1e-12)
 
    return float(clamp01(r)), {
        "obs": float(M), "base": float(M0), "sd": float(sd0),
        "n_real_close": n_real_close, "n_syn_close": n_syn_close,
        "n_bins": int(n_bins), "upper": float(upper),
    }

def internal_kinship_excess(G_syn: np.ndarray, G_ho: np.ndarray, G_tr: np.ndarray, theta_list: Iterable[float] = (0.10, 0.125, 0.25), n_boot: int = 100, random_seed: int = 123,) -> Tuple[float, Dict[float, float]]:
    """Detect excess close-kin pairs in a synthetic cohort relative to holdout.
 
    Parameters
    ----------
    G_syn : np.ndarray
        Synthetic genotype matrix.
    G_ho : np.ndarray
        Holdout genotype matrix. 
    G_tr : np.ndarray
        Training genotype matrix. Used only for AF standardisation.
    theta_list : iterable of float, default=(0.10, 0.125, 0.25)
        Kinship thresholds to evaluate in the GRM scale. Multiple
        thresholds are checked because different degrees of synthetic
        kinship excess may be most visible at different cutoffs.
    n_boot : int, default=200
        Number of bootstrap replicates for the null.
    random_seed : int, default=123
        Random seed.
 
    Returns
    -------
    ike_score : float
        Overall IKE risk score in [0, 1]: the maximum per-threshold
        risk score across all values in ``theta_list``. Returns ``0.0``
        if either GRM has no pairwise values.
    breakdown : dict of float -> float
        Per-threshold risk scores keyed by each value in ``theta_list``,
        for diagnostic inspection of which threshold(s) drove the score.
    """
    rng = np.random.default_rng(random_seed)
 
    # Compute GRMs from genotypes standardised by training AFs
    X_syn, _ = standardize_by_af(G_syn, G_tr)
    X_ho,  _ = standardize_by_af(G_ho,  G_tr)
    K_syn = compute_grm(X_syn)
    K_ho  = compute_grm(X_ho)
 
    iu_syn = np.triu_indices_from(K_syn, 1)
    iu_ho  = np.triu_indices_from(K_ho,  1)
    v_syn   = np.asarray(K_syn[iu_syn], dtype=float)
    v0_full = np.asarray(K_ho[iu_ho],   dtype=float)
 
    if v_syn.size == 0 or v0_full.size == 0:
        return 0.0, {float(t): 0.0 for t in theta_list}
 
    def frac_ge(v: np.ndarray, t: float) -> float:
        return float(np.mean(v >= t))
 
    # Genotype-level bootstrap null: compute fracs once per replicate,
    # shared across all thresholds to avoid redundant GRM computation
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
 
    r_list:    list[float]         = []
    breakdown: Dict[float, float]  = {}
 
    for theta in theta_list:
        f  = frac_ge(v_syn, theta)
        f0 = float(np.mean(boot_fracs[float(theta)]))
        r  = 0.0 if f <= f0 else (f - f0) / (1.0 - f0 + 1e-12)
        r  = float(clamp01(r))
        r_list.append(r)
        breakdown[float(theta)] = r
 
    return (max(r_list) if r_list else 0.0), breakdown

def hap_collision_rate(G: np.ndarray, var_chr: Iterable, *, window_k: int = 8, stride: int = 4, min_poly: int = 6) -> Tuple[float, float]:
    """Compute the micro-haplotype collision rate across sliding windows.
 
    Parameters
    ----------
    G : np.ndarray
        Genotype matrix.
    var_chr : iterable
        Per-SNP chromosome labels in the same column order as ``G``.
    window_k : int, default=8
        Number of consecutive SNPs per window.
    stride : int, default=4
        Step size between consecutive window start positions.
    min_poly : int, default=6
        Minimum number of polymorphic SNP columns (non-zero std) required
        within a window for it to be included. Windows with fewer
        polymorphic sites are skipped to avoid spurious collisions from
        near-monomorphic regions.
 
    Returns
    -------
    mean_rho : float
        Mean collision rate across all qualifying windows. ``0.0`` if
        no qualifying windows were found.
    max_rho : float
        Maximum collision rate observed across any single qualifying
        window. ``0.0`` if no qualifying windows were found.
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

def hap_collision_risk(G_syn: np.ndarray, var_chr: Iterable, G_ho: np.ndarray, *, window_k: int = 8, stride: int = 4, min_poly: int = 6, n_boot: int = 100, random_seed: int = 123) -> Tuple[float, Dict[str, float]]:
    """Compute a bootstrap-calibrated micro-haplotype collision risk score.

    Parameters
    ----------
    G_syn : np.ndarray
        Synthetic genotype matrix.
    var_chr : iterable
        Per-SNP chromosome labels in column order of ``G_syn`` and
        ``G_ho``.
    G_ho : np.ndarray
        Holdout genotype matrix.
    window_k : int, default=8
        Window size in SNPs.
    stride : int, default=4
        Window stride.
    min_poly : int, default=6
        Minimum polymorphic sites per window.
    n_boot : int, default=200
        Number of bootstrap replicates.
    random_seed : int, default=123
       Random seed.
 
    Returns
    -------
    risk : float
        Calibrated HAP collision risk in [0, 1]. Returns ``0.0`` if
        the holdout has fewer than 2 individuals.
    info : dict
        Diagnostic dictionary with keys:
 
        - "obs": observed mean collision rate ``c`` in ``G_syn``.
        - "base": bootstrap null mean collision rate ``c0``.
        - "cmax": observed max window collision rate in ``G_syn``,
          used as the denominator in the risk calibration.
    """
    rng = np.random.default_rng(random_seed)

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

def spectral_inflation(K: np.ndarray) -> float:
    """Compute the spectral inflation factor of a GRM.
 
    Parameters
    ----------
    K : np.ndarray
        Square symmetric GRM.
 
    Returns
    -------
    float
        Spectral inflation factor.
        Returns ``0.0`` if ``K`` is empty.
    """
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    if n == 0:
        return 0.0

    w = np.linalg.eigvalsh(K)
    lam1 = float(w[-1])
    tr = float(np.trace(K))
    return lam1 / (tr + 1e-12)

def spectral_risk(K_syn: np.ndarray, K_base: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Compare spectral inflation of a synthetic GRM against a baseline.
 
    Parameters
    ----------
    K_syn : np.ndarray
        Square symmetric GRM for the synthetic cohort.
    K_base : np.ndarray
        Square symmetric GRM for the baseline.
 
    Returns
    -------
    risk : float
        Spectral risk score in [0, 1].
    info : dict
        Diagnostic dictionary with keys:
 
        - ``"obs"``: observed spectral inflation ``s`` of ``K_syn``.
        - ``"base"``: baseline spectral inflation ``s0`` of ``K_base``.
    """
    K_syn = _symmetrize(np.asarray(K_syn, dtype=float))
    s = spectral_inflation(K_syn)

    K_base = _symmetrize(np.asarray(K_base, dtype=float))
    s0 = spectral_inflation(K_base)
    r = 0.0 if s <= s0 else (s - s0) / (s + 1e-12)
    return clamp01(r), dict(obs=s, base=s0)

def compute_kri(G_tr, G_ho, G_syn, var_chr, theta: float = 0.125, replay_n_bins: int = 25, replay_upper: float = 0.5, ike_thetas: Iterable[float] = (0.10, 0.125, 0.25), n_boot: int = 100, random_seed: int = 123, agg: str = "max") -> Dict:
    """Compute the full Kinship Risk Index (KRI) for a synthetic cohort.
 
    Parameters
    ----------
    G_tr : np.ndarray
        Training genotype matrix. Used only for
        AF standardization.
    G_ho : np.ndarray
        Holdout genotype matrix.
    G_syn : np.ndarray
        Synthetic genotype matrix to evaluate.
    var_chr : iterable
        Per-SNP chromosome labels in column order of all three matrices,
        used by the HAP collision sub-score.
    theta : float, default=0.125
        Kinship threshold for the Replay and IKE sub-scores, in the
        ``compute_grm`` output scale.
    replay_n_bins : int, default=25
        Histogram bin count for the Replay sub-score.
    replay_upper : float, default=0.5
        Upper clip value for the Replay close-kin histogram.
    ike_thetas : iterable of float, default=(0.10, 0.125, 0.25)
        Kinship thresholds for the IKE sub-score. Multiple thresholds
        are evaluated and the worst-case is taken.
    n_boot : int, default=200
        Number of bootstrap replicates for all sub-scores.
    random_seed : int, default=123
        Random seed.
    agg : str, default="max"
        Aggregation method for combining the four sub-scores
        (``r_replay``, ``r_IKE``, ``r_HAP``, ``r_SPEC``) into the
        final KRI. One of "max", "mean", or "median".
 
    Returns
    -------
    dict
        Result dictionary with the following keys:
 
        - "r_replay" (float): Replay sub-score in [0, 1].
        - "r_IKE" (float): Internal Kinship Excess sub-score in [0, 1].
        - "r_HAP" (float): Micro-haplotype collision sub-score in [0, 1].
        - "r_SPEC" (float): Spectral inflation sub-score in [0, 1].
        - "KRI" (float): Aggregate Kinship Risk Index in [0, 1],
          computed by applying ``agg`` to
          ``[r_replay, r_IKE, r_HAP, r_SPEC]``.
        - "debug" (dict): Full diagnostic sub-dictionaries from each
          sub-score ("replay", "ike_breakdown", "hap",
          "spec"), plus a "params" entry echoing all input
          hyperparameters for reproducibility.
    """
    # standardize by training AF
    X_tr, _  = standardize_by_af(G_tr, G_tr)
    X_ho, _  = standardize_by_af(G_ho, G_tr)
    X_syn, _ = standardize_by_af(G_syn, G_tr)

    K_ho  = compute_grm(X_ho)
    K_syn = compute_grm(X_syn)

    # Replay: close-kin tail distribution similarity
    # G_ho and G_tr passed for genotype-level bootstrap null
    r_replay, replay_info = replay_risk(
        G_ho=G_ho, G_syn=G_syn, G_tr=G_tr,
        theta=theta, n_bins=replay_n_bins, upper=replay_upper,
        n_boot=n_boot, random_seed=random_seed,
    )

    # Internal kinship excess
    # G_ho and G_tr passed for genotype-level bootstrap null
    r_IKE, ike_breakdown = internal_kinship_excess(
        G_syn=G_syn, G_ho=G_ho, G_tr=G_tr,
        theta_list=ike_thetas, n_boot=n_boot, random_seed=random_seed,
    )

    # Micro-haplotype collisions
    r_HAP, hap_info = hap_collision_risk(G_syn, var_chr, G_ho=G_ho, window_k=8, stride=4, min_poly=6, n_boot=n_boot, random_seed=random_seed)

    # Spectral anomaly
    r_SPEC, spec_info = spectral_risk(K_syn, K_ho)

    # Aggregate KRI
    KRI = _aggregate([r_replay, r_IKE, r_HAP, r_SPEC], agg)

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
                "random_seed":          int(random_seed),
                "agg":           str(agg),
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

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter

from score import clamp01, r_raw, calibrate_to_prism, fit_weights_anchor

from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable

plt.rcParams.update({"figure.dpi": 160, "font.size": 12})

def rank_from_scores(scores, names):
    """Sort candidate names from highest to lowest score.
 
    Parameters
    ----------
    scores : sequence of float
        Numeric scores for each candidate.
    names : sequence of str
        Candidate identifiers corresponding to each score.
 
    Returns
    -------
    list of str
    """
    order = np.argsort(-np.asarray(scores))
    return [names[i] for i in order]

def kendall_tau_a(rank1_names, rank2_names):
    """Compute Kendall's coefficient between two ranked orderings of the same items.
 
    Parameters
    ----------
    rank1_names : list of str
        Item names in the order defined by ranking 1.
    rank2_names : list of str
        Item names in the order defined by ranking 2.
    Returns
    -------
    float
        Kendall's rank coefficient.

    """
    idx1 = {n:i for i,n in enumerate(rank1_names)}
    idx2 = {n:i for i,n in enumerate(rank2_names)}
    names = list(idx1.keys())
    C = D = 0
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a,b = names[i], names[j]
            s1 = np.sign(idx1[a]-idx1[b]); s2 = np.sign(idx2[a]-idx2[b])
            if s1*s2 > 0: C += 1
            elif s1*s2 < 0: D += 1
    denom = len(names)*(len(names)-1)/2
    return (C-D)/denom if denom>0 else 0.0

def p_fmt_pow10(p, digits=1, n_boot=None):
    """Format a p-value as a scientific-notation string with a base-10 exponent.
 
    Parameters
    ----------
    p : float
        The p-value to format. 
    digits : int, default=1
        Number of decimal places to show.
    n_boot : int, optional
        Number of bootstrap replicates used to compute p. When p
        is exactly 0.0, the minimum detectable p-value is approximately
        1/B, so the returned string is formatted as "<1x10^-a".
 
    Returns
    -------
    str
        The formatted string.
    """
    if p <= 0.0:
        if n_boot is None:
            return "<1x10^-∞"
        a = math.ceil(math.log10(float(n_boot)))  # min detectable ~ 1/B
        return f"<1x10^-{a}"
    # use scientific notation then reformat
    m_str, e_str = f"{p:.{digits+1}e}".split("e")  # one extra digit for rounding
    m = float(m_str)
    a = int(e_str)  # typically negative
    return f"{m:.{digits}f}x10^{a}"

def bootstrap_analysis(df, n_boot=100, sigma=0.01, eps=0.02, lam=0.80, gamma=1e-3, w0=[0,0,0], refit_weights=False, step=0.01, random_seed=123):
    """Assess PRISM-G score stability and pairwise significance via bootstrapping.
 
    Parameters
    ----------
    df : pd.DataFrame
        Input table with columns ``name``, ``role``, ``pli``, ``kri``,
        ``tli``. ``role`` must be one of ``"safe anchor"``,
        ``"leaky anchor"``, or ``"candidate"``.
    n_boot : int, default=100
        Number of bootstrap replicates.
    sigma : float, default=0.01
        Standard deviation of the Gaussian jitter applied to each anchor
        metric score on every replicate.
    eps : float, default=0.02
        Target raw risk score for the safe anchor.
    lam : float, default=0.80
        Target raw risk score for the leaky anchor.
    gamma : float, default=1e-3
        Ridge regularization strength.
    w0 : sequence of float, default=(0, 0, 0)
        Ridge regularization anchor (the point ``w`` is penalized toward).
    refit_weights : bool, default=False
        If ``False`` (default), weights are fitted once on the un-jittered
        anchor scores and reused for all ``n_boot`` replicates. This isolates
        score variability, holding the weight vector
        fixed. If ``True``, weights are refitted on every replicate, so both weight uncertainty and score
        uncertainty are propagated; the returned ``weights_out`` is then a
        dict with keys ``"mean"``, ``"sd"``, and ``"n"`` summarizing the
        distribution of fitted weights across replicates.
    step : float, default=0.01
        Grid resolution.
    random_seed : int, default=123
        Random seed.

    Returns
    -------
    weights_out : np.ndarray or dict
        If ``refit_weights=False``: the single fixed weight vector
        ``[w_PLI, w_KRI, w_TLI]`` used across all replicates.
        If ``refit_weights=True``: a dict with keys "mean",
        "sd", and "n", summarizing the per-replicate
        fitted weights.
    tbl_stats_R : pd.DataFrame
        Per-candidate summary of raw ``R`` score distributions across
        bootstrap replicates, sorted descending by ``mean_R``. Columns:
        ``name``, ``mean_R``, ``sd_R``, ``ci95_R_lo``, ``ci95_R_hi``.
    tbl_stats_G : pd.DataFrame
        Per-candidate summary of calibrated PRISM-G score distributions,
        sorted descending by ``mean_G``. Columns: ``name``, ``mean_G``,
        ``sd_G``, ``ci95_G_lo``, ``ci95_G_hi``.
    tbl_pvals : pd.DataFrame
        All pairwise comparisons of PRISM-G scores (A vs B for every pair
        of candidates), sorted ascending by ``p_two_sided``. Columns:
        ``A_minus_B``, ``mean_diff``, ``ci95_diff_lo``, ``ci95_diff_hi``,
        ``p_two_sided``, ``p_fmt``.
    rank_strings : list of str
        The full ranking string for each bootstrap replicate, formatted as
        "name_A > name_B > ..." from highest to lowest PRISM-G score.
    kendall_summary : dict
        Summary of Kendall tau-a ranking stability. Keys:
 
        - "baseline": the ranking string of the first replicate,
          used as the reference ordering for all tau computations.
        - "mean_tau", "sd_tau": mean and SD of tau across
          all n_boot replicates relative to the baseline.
        - "tau_ci95": ``[lo, hi]`` 2.5/97.5 percentile interval.
        - "perm_null_mean", "perm_null_sd": mean and SD of tau
          under a permutation null.
        - "p_value_two_sided": two-sided permutation p-value
          comparing ``mean_tau`` against the null distribution.
        - "p_fmt": formatted p-value string from ``p_fmt_pow10``.
    boot_G : dict of str -> list of float
        Raw per-replicate PRISM-G score arrays keyed by candidate name.

    """
    rng = np.random.default_rng(random_seed)
    df = df.copy()
    idx_safe = df.index[df["role"]=="safe anchor"][0]
    idx_leak = df.index[df["role"]=="leaky anchor"][0]
    cand_names = df.loc[df["role"]=="candidate","name"].tolist()

    r_safe0 = df.loc[idx_safe, ["pli","kri","tli"]].values.astype(float)
    r_leak0 = df.loc[idx_leak, ["pli","kri","tli"]].values.astype(float)
    w0 = np.asarray(w0, float)

    weights_fixed = None
    if not refit_weights:
        weights_fixed = fit_weights_anchor(r_safe0, r_leak0, eps=eps, lam=lam, gamma=gamma, step=step, w0=w0)

    boot_R = {n: [] for n in cand_names}
    boot_G = {n: [] for n in cand_names}
    rank_strings = []
    w_history = []

    def jitter(v): return np.clip(v + rng.normal(0, sigma), 0.0, 1.0)

    for _ in range(n_boot):
        r_safe = np.array([jitter(x) for x in r_safe0], float)
        r_leak = np.array([jitter(x) for x in r_leak0], float)
        w = fit_weights_anchor(r_safe, r_leak, eps=eps, lam=lam, gamma=gamma, step=step, w0=w0) if refit_weights else weights_fixed
        if refit_weights: w_history.append(w)

        alpha = r_raw(r_safe, w); beta = r_raw(r_leak, w)

        scores_G = []
        for _, row in df[df["role"]=="candidate"].iterrows():
            r = row[["pli","kri","tli"]].values.astype(float)
            Rb = r_raw(r, w)
            Gb = calibrate_to_prism(Rb, alpha, beta)
            boot_R[row["name"]].append(Rb)
            boot_G[row["name"]].append(Gb)
            scores_G.append(Gb)
        rank_strings.append(" > ".join(rank_from_scores(scores_G, cand_names)))

    def ci95(a):
        lo, hi = np.percentile(a, [2.5,97.5]); return float(lo), float(hi)

    rowsR, rowsG = [], []
    for name in cand_names:
        arrR = np.array(boot_R[name]); arrG = np.array(boot_G[name])
        rowsR.append({"name": name, "mean_R": arrR.mean(), "sd_R": arrR.std(ddof=1),
                      "ci95_R_lo": ci95(arrR)[0], "ci95_R_hi": ci95(arrR)[1]})
        rowsG.append({"name": name, "mean_G": arrG.mean(), "sd_G": arrG.std(ddof=1),
                      "ci95_G_lo": ci95(arrG)[0], "ci95_G_hi": ci95(arrG)[1]})
    tbl_stats_R = pd.DataFrame(rowsR).sort_values("mean_R", ascending=False).reset_index(drop=True)
    tbl_stats_G = pd.DataFrame(rowsG).sort_values("mean_G", ascending=False).reset_index(drop=True)

    # pairwise PRISM-G differences, two-sided p-values
    rowsP = []
    for A_name, B_name in combinations(cand_names, 2):
        d = np.array(boot_G[A_name]) - np.array(boot_G[B_name])
        p_two = float(min(1.0, 2 * min(np.mean(d >= 0), np.mean(d <= 0))))
        lo, hi = np.percentile(d, [2.5, 97.5])
        rowsP.append({
            "A_minus_B": f"{A_name} - {B_name}",
            "mean_diff": float(d.mean()),
            "ci95_diff_lo": float(lo),
            "ci95_diff_hi": float(hi),
            "p_two_sided": p_two,
            "p_fmt": p_fmt_pow10(p_two, digits=1, n_boot=n_boot)
        })
    tbl_pvals = pd.DataFrame(rowsP).sort_values("p_two_sided").reset_index(drop=True)

    # Kendall τ on bootstrap rank strings
    baseline = rank_strings[0]
    taus = np.array([kendall_tau_a(rs.split(" > "), baseline.split(" > ")) for rs in rank_strings])
    rng2 = np.random.default_rng(random_seed)
    perms = [" > ".join(rng2.permutation(cand_names)) for _ in range(n_boot)]
    taus_null = np.array([kendall_tau_a(p.split(" > "), baseline.split(" > ")) for p in perms])
    p_tau = 2 * min(np.mean(taus_null >= np.mean(taus)), np.mean(taus_null <= np.mean(taus)))
    kendall_summary = {
        "baseline": baseline,
        "mean_tau": float(np.mean(taus)),
        "sd_tau": float(np.std(taus, ddof=1)),
        "tau_ci95": [float(np.percentile(taus,2.5)), float(np.percentile(taus,97.5))],
        "perm_null_mean": float(np.mean(taus_null)),
        "perm_null_sd": float(np.std(taus_null, ddof=1)),
        "p_value_two_sided": float(min(1.0, p_tau)),
        "p_fmt": p_fmt_pow10(float(min(1.0, p_tau)), digits=1, n_boot=n_boot)
    }

    weights_out = ({"mean": np.mean(w_history,0), "sd": np.std(w_history,0,ddof=1), "n": len(w_history)}
                   if refit_weights else weights_fixed)
    return weights_out, tbl_stats_R, tbl_stats_G, tbl_pvals, rank_strings, kendall_summary, boot_G

def grid_search(df, eps_grid=(0.02, 0.04, 0.06, 0.08, 0.10), lam_grid=(0.75, 0.80, 0.85, 0.90), gamma_grid=(1e-4, 1e-3, 1e-2), *, n_boot=100, sigma=0.01, w0=(0,0,0), step=0.01, refit_weights=False, random_seed=123):
    """Grid search evaluation: sweep over (eps, lam, gamma) configurations and rank them by ranking stability.
 
    Parameters
    ----------
    df : pd.DataFrame
        Input table passed directly to ``bootstrap_analysis``. Must contain
        columns ``name``, ``role``, ``pli``, ``kri``, ``tli`` with exactly
        one ``"safe anchor"`` and one ``"leaky anchor"`` row.
    eps_grid : sequence of float, default=(0.02, 0.04, 0.06, 0.08, 0.10)
        Values of ``eps`` (safe anchor target raw score) to sweep over.
    lam_grid : sequence of float, default=(0.75, 0.80, 0.85, 0.90)
        Values of ``lam`` (leaky anchor target raw score) to sweep over.
    gamma_grid : sequence of float, default=(1e-4, 1e-3, 1e-2)
        Values of ``gamma`` (ridge regularization strength) to sweep over.
    n_boot : int, default=100
        Number of bootstrap replicates per configuration.
    sigma : float, default=0.01
        Jitter standard deviation.
    w0 : sequence of float, default=(0, 0, 0)
        Ridge regularization initial weights.
    step : float, default=0.01
        Weight grid resolution.
    refit_weights : bool, default=True
        Whether to refit weights on every bootstrap replicate. Defaults
        to ``True`` here (unlike ``bootstrap_analysis``'s default of
        ``False``) so the grid search reflects weight uncertainty as well
        as score uncertainty. When ``True``, the output DataFrame includes
        per-metric weight mean and SD columns.
    random_seed : int, default=123
        Random seed.
 
    Returns
    -------
    pd.DataFrame
        One row per (eps, lam, gamma) combination, sorted descending by
        ``(mean_tau, modal_prop)``. Columns:
 
        - ``eps``, ``lam``, ``gamma``: the hyperparameter values for this
          configuration.
        - ``mean_tau``, ``sd_tau``: mean and SD of Kendall tau-a across
          bootstrap replicates relative to the baseline ranking.
        - ``tau_ci_lo``, ``tau_ci_hi``: 95% CI for tau.
        - ``p_tau``: two-sided permutation p-value for the Kendall tau
          test. 
        - ``modal_rank``: the most frequently occurring ranking string
          ("A > B > C > ...") across all ``n_boot`` replicates.
        - ``modal_prop``: fraction of replicates that produced
          ``modal_rank``.
        - ``N_boot``: number of bootstrap replicates (always equals ``n_boot``).
        - ``w_PLI_mean``, ``w_KRI_mean``, ``w_TLI_mean``,
          ``w_PLI_sd``, ``w_KRI_sd``, ``w_TLI_sd`` *(only when*
          ``refit_weights=True`` *)*: mean and SD of each fitted weight
          across bootstrap replicates.
    """
    rows = []
    for gam in gamma_grid:
        for eps in eps_grid:
            for lam in lam_grid:
                weights_info, tblR, tblG, tblP, ranks, kend, boot_G = bootstrap_analysis(
                    df, n_boot=n_boot, sigma=sigma, eps=eps, lam=lam, gamma=gam, w0=w0,
                    refit_weights=refit_weights, step=step, random_seed=random_seed
                )

                # modal ranking across bootstraps
                counts = Counter(ranks)
                modal_rank, modal_freq = counts.most_common(1)[0]
                modal_prop = modal_freq / len(ranks)

                # pull Kendall summary
                mean_tau   = kend["mean_tau"]
                sd_tau     = kend["sd_tau"]
                ci_lo, ci_hi = kend["tau_ci95"]
                p_tau      = kend["p_value_two_sided"]

                row = {
                    "eps": eps, "lam": lam, "gamma": gam,
                    "mean_tau": mean_tau, "sd_tau": sd_tau,
                    "tau_ci_lo": ci_lo, "tau_ci_hi": ci_hi,
                    "p_tau": p_tau,
                    #"p_tau_10^-a": p_to_pow10_nearest(p_tau, B), TO CHANGE
                    "modal_rank": modal_rank,
                    "modal_prop": modal_prop,
                    "N_boot": len(ranks)
                }
                # if refitting weights, also emit their mean/sd (optional)
                if isinstance(weights_info, dict):
                    row.update({
                        "w_PLI_mean": float(weights_info["mean"][0]),
                        "w_KRI_mean": float(weights_info["mean"][1]),
                        "w_TLI_mean": float(weights_info["mean"][2]),
                        "w_PLI_sd":   float(weights_info["sd"][0]),
                        "w_KRI_sd":   float(weights_info["sd"][1]),
                        "w_TLI_sd":   float(weights_info["sd"][2]),
                    })
                rows.append(row)

    res = pd.DataFrame(rows).sort_values(
        ["mean_tau", "modal_prop"], ascending=[False, False]
    ).reset_index(drop=True)
    return res

__all__ = ["rank_from_scores", 
         "kendall_tau_a",
         "bootstrap_analysis",
         "grid_search",]
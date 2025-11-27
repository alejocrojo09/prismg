import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter

from prismg.score import clamp01, r_raw, calibrate_to_prism

from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable

plt.rcParams.update({"figure.dpi": 160, "font.size": 12})

def rank_from_scores(scores, names):
    order = np.argsort(-np.asarray(scores))
    return [names[i] for i in order]

def kendall_tau_a(rank1_names, rank2_names):
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

def _entropy_safe(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    p_pos = p[p > 0.0]
    if p_pos.size == 0:
        return 0.0
    return float(-(p_pos * np.log(p_pos)).sum())

def fit_weights_anchor(r_safe, r_leak, eps=0.02, lam=0.80, gamma=1e-3, step=0.01, w0=None, tol=1e-12):
    if w0 is None:
        w0 = np.array([0, 0, 0], float)
    w0 = np.asarray(w0, float)

    best_loss, best_w = np.inf, np.array([0, 0, 0], float)
    grid = np.arange(0, 1 + 1e-12, step)

    for w1 in grid:
        for w2 in grid:
            w3 = 1 - w1 - w2
            if w3 < 0:
                continue
            w = np.array([w1, w2, w3], float)
            loss = (r_raw(r_safe, w) - eps) ** 2 + (r_raw(r_leak, w) - lam) ** 2 + gamma * np.sum((w - w0) ** 2)

            if loss + tol < best_loss:
                best_loss, best_w = loss, w
            elif abs(loss - best_loss) <= tol:
                if np.sum((w - w0) ** 2) + 1e-15 < np.sum((best_w - w0) ** 2):
                    best_w = w
                else:
                    if _entropy_safe(w) > _entropy_safe(best_w) + 1e-12:
                        best_w = w
    return best_w

def p_fmt_pow10(p, digits=1, B=None):
    if p <= 0.0:
        if B is None:
            return "<1x10^-∞"
        a = math.ceil(math.log10(float(B)))  # min detectable ~ 1/B
        return f"<1x10^-{a}"
    # use scientific notation then reformat
    m_str, e_str = f"{p:.{digits+1}e}".split("e")  # one extra digit for rounding
    m = float(m_str)
    a = int(e_str)  # typically negative
    return f"{m:.{digits}f}x10^{a}"

def bootstrap_analysis(df, B=1000, sigma=0.01, eps=0.02, lam=0.80, gamma=1e-3, w0=(0,0,0), refit_weights=False, step=0.01, random_state=123):
    rng = np.random.default_rng(random_state)
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

    for _ in range(B):
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
            "p_fmt": p_fmt_pow10(p_two, digits=1, B=B)
        })
    tbl_pvals = pd.DataFrame(rowsP).sort_values("p_two_sided").reset_index(drop=True)

    # Kendall τ on bootstrap rank strings
    baseline = rank_strings[0]
    taus = np.array([kendall_tau_a(rs.split(" > "), baseline.split(" > ")) for rs in rank_strings])
    rng2 = np.random.default_rng(777)
    perms = [" > ".join(rng2.permutation(cand_names)) for _ in range(2000)]
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
        "p_fmt": p_fmt_pow10(float(min(1.0, p_tau)), digits=1, B=B)
    }

    weights_out = ({"mean": np.mean(w_history,0), "sd": np.std(w_history,0,ddof=1), "n": len(w_history)}
                   if refit_weights else weights_fixed)
    return weights_out, tbl_stats_R, tbl_stats_G, tbl_pvals, rank_strings, kendall_summary, boot_G

def grid_search(df, eps_grid=(0.02, 0.04, 0.06, 0.08, 0.10), lam_grid=(0.75, 0.80, 0.85, 0.90), gamma_grid=(1e-4, 1e-3, 1e-2), *, B=600, sigma=0.01, w0=(0,0,0), step=0.01, refit_weights=True, random_state=123):
    rows = []
    for gam in gamma_grid:
        for eps in eps_grid:
            for lam in lam_grid:
                weights_info, tblR, tblG, tblP, ranks, kend, boot_G = bootstrap_analysis(
                    df, B=B, sigma=sigma, eps=eps, lam=lam, gamma=gam, w0=w0,
                    refit_weights=refit_weights, step=step, random_state=random_state
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
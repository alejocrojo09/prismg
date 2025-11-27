from __future__ import annotations
from typing import Dict, Iterable, Tuple

import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

def train_maf(G_tr, alpha: float = 0.0):
    G_tr  = np.asarray(G_tr, float)
    nobs  = np.sum(~np.isnan(G_tr), axis=0)
    alt   = np.nansum(G_tr, axis=0)
    if alpha and alpha > 0:
        denom = 2.0 * nobs + 2.0 * alpha
        p = (alt + alpha) / np.where(denom > 0, denom, 1.0)
    else:
        p = np.where(nobs > 0, alt / (2.0 * nobs), 0.0)
    p   = np.clip(p, 1e-12, 1 - 1e-12)
    maf = np.minimum(p, 1 - p)
    return p, maf, nobs

def rare_burden_features(G, rare_mask, var_chr):
    chroms = [c for c in sorted({str(x) for x in var_chr}) if c.isdigit() and 1 <= int(c) <= 22]
    idx_by_chr = {c: np.where(rare_mask & (np.array(var_chr) == c))[0] for c in chroms}
    feats = [np.nansum(G[:, idx_by_chr[c]], axis=1) if len(idx_by_chr[c]) else np.zeros(G.shape[0])
             for c in chroms]
    feats.append(np.nansum(G[:, rare_mask], axis=1) if rare_mask.any() else np.zeros(G.shape[0]))
    X = np.vstack(feats).T
    names = [f"rare_chr{c}" for c in chroms] + ["rare_global"]
    return X, names

def r_mia_exposure(C_tr, C_ho, C_syn):
    nn = NearestNeighbors(n_neighbors=1, metric="manhattan").fit(C_syn)
    d_tr = nn.kneighbors(C_tr, 1, return_distance=True)[0][:,0]
    d_ho = nn.kneighbors(C_ho, 1, return_distance=True)[0][:,0]
    y = np.r_[np.ones(len(d_tr)), np.zeros(len(d_ho))]   # 1=TRAIN, 0=HOLDOUT
    scores = np.r_[-d_tr, -d_ho]                         # closer → larger
    auc = roc_auc_score(y, scores)
    r   = max(0.0, min(1.0, 2*(auc - 0.5)))
    return r, float(auc)

def r_uniq_collision(G_syn, G_tr, rare_mask=None, maf_thresh: float = 1e-3, alpha: float = 1e-3, k_minor: int = 1, min_train_calls_frac: float = 0.8):
    p_hat, maf_hat, nobs_tr = train_maf(G_tr, alpha=alpha)

    ntr_eff   = np.maximum(nobs_tr, 1)
    maf_floor = (k_minor / (2.0 * ntr_eff)) if k_minor > 0 else 0.0
    maf_eff   = np.maximum(maf_thresh, maf_floor)

    rare = (maf_hat < maf_eff) if rare_mask is None else (rare_mask & (maf_hat < maf_eff))
    min_calls = max(1, int(min_train_calls_frac * G_tr.shape[0]))
    rare &= (nobs_tr >= min_calls)
    if rare.sum() == 0:
        return 0.0, 0.0, 0.0, {"n_rare": 0, "n_used": 0}

    S        = np.asarray(G_syn, float)[:, rare]
    valid    = ~np.isnan(S)
    carriers = (np.nan_to_num(S, nan=0.0) > 0) & valid
    n_eff    = valid.sum(axis=0)
    x_cnt    = carriers.sum(axis=0)
    ok       = (n_eff >= 2)
    if not np.any(ok):
        return 0.0, 0.0, 0.0, {"n_rare": int(rare.sum()), "n_used": 0}

    U = float(np.mean(x_cnt[ok] >= 2))

    p_r  = p_hat[rare][ok]
    q    = 1.0 - (1.0 - p_r)**2
    n_j  = n_eff[ok].astype(float)
    U0_per = 1.0 - (1.0 - q)**n_j - n_j * q * (1.0 - q)**(n_j - 1.0)
    U0   = float(np.mean(U0_per))

    r = 0.0 if U <= U0 else (U - U0) / (1.0 - U0 + 1e-12)
    r = float(np.clip(r, 0.0, 1.0))
    dbg = {"n_rare": int(rare.sum()), "n_used": int(ok.sum()),
           "U": U, "U0": U0, "mean_n_eff": float(n_j.mean()),
           "min_train_calls": int(min_calls)}
    return U, U0, r, dbg

def compute_tli(G_tr, G_ho, G_syn, var_chr, maf_thresh: float = 1e-3, alpha: float = 1e-3, k_minor: int = 1, min_train_calls_frac: float = 0.8):
    _, maf_tr, _ = train_maf(G_tr, alpha=alpha)
    n_train = max(G_tr.shape[0], 1)
    global_floor = 1.0 / (2.0 * n_train)
    rare_mask = (maf_tr < np.maximum(maf_thresh, global_floor))

    C_tr,  _ = rare_burden_features(G_tr,  rare_mask, var_chr)
    C_ho,  _ = rare_burden_features(G_ho,  rare_mask, var_chr)
    C_syn, _ = rare_burden_features(G_syn, rare_mask, var_chr)

    r_mia, auc = r_mia_exposure(C_tr, C_ho, C_syn)

    U, U0, r_uniq, dbg = r_uniq_collision(
        G_syn, G_tr,
        rare_mask=rare_mask,
        maf_thresh=maf_thresh, alpha=alpha,
        k_minor=k_minor, min_train_calls_frac=min_train_calls_frac
    )
    return {"MIA_AUC": auc, "r_mia": r_mia, "U": U, "U0": U0,
            "r_uniq": r_uniq, "TLI_final": max(r_mia, r_uniq), "dbg": dbg}

__all__ = [
    "train_maf",
    "rare_burden_features",
    "r_mia_exposure",
    "r_uniq_collision",
    "compute_tli",
]

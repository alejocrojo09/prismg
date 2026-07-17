from __future__ import annotations

import numpy as np
from prismg.score import _aggregate

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

def train_maf(G_tr, alpha: float = 0.0):
    """Estimate per-SNP allele frequencies and MAF from the training matrix.
 
    Parameters
    ----------
    G_tr : np.ndarray
        Training genotype dosage matrix, values in
        {0, 1, 2} or NaN. NaN entries are excluded from frequency
        estimation.
    alpha : float, default=0.0
        Pseudocount for Laplace smoothing. When ``alpha > 0``, the
        allele frequency is estimated as
        ``(alt_count + alpha) / (2 * n_obs + 2 * alpha)``, which
        prevents zero frequencies for variants unseen in training.
        When ``alpha=0``, the maximum-likelihood estimate is used.
 
    Returns
    -------
    p : np.ndarray
        Estimated ALT allele frequency per SNP.
    maf : np.ndarray
        Minor allele frequency per SNP.
    nobs : np.ndarray
        Number of non-NaN observations per SNP.
    """

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
    """Compute per-individual rare-variant burden features by chromosome.

    Parameters
    ----------
    G : np.ndarray
        Genotype dosage matrix.
    rare_mask : np.ndarray
        Boolean mask indicating which SNP columns are
        considered rare.
    var_chr : iterable
        Per-SNP chromosome labels in the same column order as ``G``.
    Returns
    -------
    X : np.ndarray
        Rare-burden feature matrix.
    names : list of str
        Feature names, one per column of ``X``: ``["rare_chr1",
        "rare_chr2", ..., "rare_global"]``.
    """
    chroms = [c for c in sorted({str(x) for x in var_chr}) if c.isdigit() and 1 <= int(c) <= 22]
    idx_by_chr = {c: np.where(rare_mask & (np.array(var_chr) == c))[0] for c in chroms}
    feats = [np.nansum(G[:, idx_by_chr[c]], axis=1) if len(idx_by_chr[c]) else np.zeros(G.shape[0])
             for c in chroms]
    feats.append(np.nansum(G[:, rare_mask], axis=1) if rare_mask.any() else np.zeros(G.shape[0]))
    X = np.vstack(feats).T
    names = [f"rare_chr{c}" for c in chroms] + ["rare_global"]
    return X, names

def r_mia_exposure(C_tr, C_ho, C_syn):
    """Compute a Membership Inference Attack (MIA) exposure score via rare-burden proximity.
 
    Parameters
    ----------
    C_tr : np.ndarray
        Rare-burden feature matrix for the training cohort.
    C_ho : np.ndarray
        Rare-burden feature matrix for the holdout cohort.
    C_syn : np.ndarray
        Rare-burden feature matrix for the synthetic cohort.
 
    Returns
    -------
    r_mia : float
        MIA exposure risk score in [0, 1]. Maps AUC=0.5 to 0.0 (no
        signal) and AUC=1.0 to 1.0 (maximally leaky).
    auc : float
        Raw AUC value in [0.5, 1.0] from the nearest-synthetic-neighbour
        distance discrimination test.
    """
    nn = NearestNeighbors(n_neighbors=1, metric="manhattan").fit(C_syn)
    d_tr = nn.kneighbors(C_tr, 1, return_distance=True)[0][:,0]
    d_ho = nn.kneighbors(C_ho, 1, return_distance=True)[0][:,0]
    y = np.r_[np.ones(len(d_tr)), np.zeros(len(d_ho))]   # 1=TRAIN, 0=HOLDOUT
    scores = np.r_[-d_tr, -d_ho]                         # closer → larger
    auc = roc_auc_score(y, scores)
    r   = max(0.0, min(1.0, 2*(auc - 0.5)))
    return r, float(auc)

def r_uniq_collision(G_syn, G_tr, rare_mask=None, maf_thresh: float = 1e-3, alpha: float = 1e-3, k_minor: int = 1, min_train_calls_frac: float = 0.8):
    """Test whether rare variants in the synthetic cohort are shared beyond HWE expectation.

    Parameters
    ----------
    G_syn : np.ndarray
        Synthetic genotype dosage matrix.
    G_tr : np.ndarray
        Training genotype dosage matrix. Used to
        estimate allele frequencies and define rare variants.
    rare_mask : np.ndarray, optional
        Boolean mask of shape pre-selecting candidate rare
        variants.
    maf_thresh : float, default=1e-3
        MAF threshold below which a variant is considered rare. The
        effective threshold per variant is
        ``max(maf_thresh, k_minor / (2 * n_obs))``, which avoids
        calling variants rare when there is insufficient training data
        to estimate their frequency reliably.
    alpha : float, default=1e-3
        Pseudocount for Laplace-smoothed AF estimation.
    k_minor : int, default=1
        Minimum minor allele count in training required for a variant
        to be included. Combined with ``maf_thresh`` to produce a
        per-variant effective MAF floor.
    min_train_calls_frac : float, default=0.8
        Minimum fraction of training individuals required to have a
        non-missing call at a variant for it to be included. Variants
        with too many missing calls have unreliable AF estimates.
 
    Returns
    -------
    U : float
        Observed fraction of qualifying rare variants with ≥ 2 carriers
        in the synthetic cohort.
    U0 : float
        HWE-expected fraction (analytical null). 
    r_uniq : float
        Unique collision risk score in [0, 1],
    dbg : dict
        Diagnostic dictionary with keys:
 
        - "n_rare": number of variants passing the MAF/mask filter.
        - "n_used": number of variants with ≥ 2 non-missing synthetic
          calls (used in the final U/U0 computation).
        - "U", "U0": as above.
        - "mean_n_eff": mean number of non-missing synthetic
          individuals across used variants.
        - "min_train_calls": minimum non-missing call count threshold
          applied.
    """
    
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

def compute_tli(G_tr, G_ho, G_syn, var_chr, maf_thresh: float = 1e-3, alpha: float = 1e-3, k_minor: int = 1, min_train_calls_frac: float = 0.8, agg: str = "max"):
    """Compute the full Trait Leakage Index (TLI) for a synthetic cohort.
 
    Parameters
    ----------
    G_tr : np.ndarray
        Training genotype dosage matrix. Used for
        AF estimation, rare-variant definition, and MIA feature building.
    G_ho : np.ndarray
        Holdout genotype dosage matrix. Serves as
        the MIA negative class (non-members).
    G_syn : np.ndarray
        Synthetic genotype dosage matrix to evaluate.
    var_chr : iterable
        Per-SNP chromosome labels in column order to build per-chromosome burden features.
    maf_thresh : float, default=1e-3
        MAF threshold for defining rare variants.
    alpha : float, default=1e-3
        Pseudocount for Laplace-smoothed AF estimation.
    k_minor : int, default=1
        Minimum minor allele count in training for a variant to
        qualify as rare.
    min_train_calls_frac : float, default=0.8
        Minimum fraction of training individuals with non-missing calls
        required at a variant.
    agg : str, default="max"
        Aggregation method for combining ``r_mia`` and ``r_uniq`` into
        the final TLI. One of ``"max"``, ``"mean"``, or ``"median"``.
 
    Returns
    -------
    dict
        Result dictionary with the following keys:
 
        - "MIA_AUC" (float): raw AUC from the MIA proximity test.
          0.5 = no signal, 1.0 = maximally leaky.
        - "r_mia" (float): MIA exposure sub-score in [0, 1].
        - "U" (float): observed fraction of rare variants with ≥ 2
          carriers in the synthetic cohort.
        - "U0" (float): HWE-expected fraction (analytical null).
        - "r_uniq" (float): unique collision sub-score in [0, 1].
        - "TLI" (float): aggregate Trait Leakage Index in [0, 1],
          computed by applying ``agg`` to ``[r_mia, r_uniq]``.
        - "dbg" (dict): diagnostic output from ``r_uniq_collision``.
 
    """
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
            "r_uniq": r_uniq, "TLI": _aggregate([r_mia, r_uniq], agg), "dbg": dbg}

__all__ = [
    "train_maf",
    "rare_burden_features",
    "r_mia_exposure",
    "r_uniq_collision",
    "compute_tli",
]

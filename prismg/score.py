def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def or_like_raw(pli, kri, tli, weights=None):
    """
    R_raw = 1 - Π_m (1 - w_m * r_m), m in {PLI, KRI, TLI}, sum w_m = 1
    Drops missing components (None/NaN) and renormalizes weights.
    """
    comps = {"PLI": pli, "KRI": kri, "TLI": tli}
    if weights is None:
        weights = {"PLI": 1/3, "KRI": 1/3, "TLI": 1/3}

    # keep only available components
    kept = {m: comps[m] for m in comps if comps[m] is not None and not np.isnan(comps[m])}
    if not kept:
        return 0.0

    # renormalize weights to components used
    w = np.array([weights[m] for m in kept])
    w = w / w.sum()

    r = np.array(list(kept.values()), dtype=float)
    return float(1.0 - np.prod(1.0 - w * r))

def calibrate_to_prism(R_raw, alpha, beta):
    """PRISM-G = 100 * clamp((R_raw - alpha) / (beta - alpha), 0, 1)."""
    eps = 1e-12
    return 100.0 * clamp01((R_raw - alpha) / (max(beta - alpha, eps)))

def median_anchor_Rraw(rows, weights=None):
    """rows = iterable of dicts with keys pli/kri/tli."""
    vals = [or_like_raw(r["pli"], r["kri"], r["tli"], weights) for r in rows]
    return float(np.median(vals)) if len(vals) else 0.0
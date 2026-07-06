import numpy as np

def _aggregate(values: list, agg: str) -> float:
    """Aggregate a list of sub-scores into a single float.
 
    Parameters
    ----------
    values : list of float
        Sub-scores to aggregate, all in [0, 1].
    agg : str
        Aggregation method. One of:
 
        - ``"max"`` (default): conservative worst-case.
        - ``"mean"``: average of all sub-scores.
        - ``"median"``: median of all sub-scores.
        
    Returns
    -------
    float
        The aggregated score.
    """
    if agg == "mean":
        return float(np.mean(values))
    if agg == "median":
        return float(np.median(values))
    return float(max(values))  # default: max

def clamp01(x):
    """Clamp a single numeric value into the closed interval [0, 1].
 
    Parameters
    ----------
    x : float
        Value to clamp. 
 
    Returns
    -------
    float
        ``x`` clamped to the range [0.0, 1.0]. Values below 0 become 0.0,
        values above 1 become 1.0, and values already in range are
        returned unchanged.
    """
    return max(0.0, min(1.0, float(x)))

def r_raw(r_vec, w_vec):
    """Combine per-metric risk scores into a single raw risk value via noisy-OR.

    Parameters
    ----------
    r_vec : list
        Component risk scores
    w_vec : list
        Per-metric weights, same length as ``r_vec``. If omitted, all
        metrics are weighted equally. 

    Returns
    -------
    float
        The combined raw risk score. 

    Raises
    ------
    ValueError
        If ``w_vec`` is provided and its length does not match ``r_vec``.
 
    """
    r = np.array([np.nan if v is None else float(v) for v in r_vec], dtype=float)
    keep = ~np.isnan(r)
    if not np.any(keep):
        return 0.0

    
    w = np.array([float(x) for x in w_vec], dtype=float)
    if w.shape != r.shape:
        raise ValueError("w_vec must have the same length as r_vec")

    r = r[keep]
    w = w[keep]

    s = w.sum()
    if s <= 0:
        # Explicitly allow the zero-weight edge case (result = 0.0)
        return 0.0
    w = w / s

    return float(1.0 - np.prod(1.0 - w * r))

def calibrate_to_prism(R_raw: float, alpha: float, beta: float, eps: float = 1e-12) -> float:
    """Rescale a raw risk value onto a 0-100 PRISM score using [alpha, beta] as anchors.
 
    Parameters
    ----------
    R_raw : float
        The raw combined risk score to calibrate.
    alpha : float
        Raw score corresponding to the safe anchor; maps to 0.
    beta : float
        Raw score corresponding to leaky anchor; maps to 100.
    eps : float, default=1e-12
        Minimum allowed denominator, used to avoid division by zero.

    Returns
    -------
    float
        The calibrated PRISM score, mapped to [0, 100].
    """
    denom = max(beta - alpha, eps)
    return 100.0 * clamp01((R_raw - alpha) / denom)

def fit_weights_anchor(r_safe, r_leak, eps=0.05, lam=0.8, gamma=1e-3, w0 = [0,0,0], step=0.01):
    """Ridge (L2) regularization using reference anchors.
 
    Searches over 3-component weight vector on a regular grid, and picks the
    weights that minimize the loss. 
 
    Parameters
    ----------
    r_safe : sequence of float or None
        Per-metric risk scores for the safe anchor.
    r_leak : sequence of float or None
        Per-metric risk scores for the leaky anchor.
    eps : float, default=0.05
        Target combined raw risk score for the safe anchor.
    lam : float, default=0.8
        Target combined raw risk score for the leaky anchor.
    gamma : float, default=1e-3
        Strength of the ridge (L2) regularization term.
    w0: sequence of float or None. 
        Initial weight seed. If none, w0 = [0,0,0]
    step : float, default=0.01
        Grid resolution for ``w1``/``w2`` in [0, 1]; ``w3`` is derived as
        ``1 - w1 - w2`` and any grid point with ``w3 < 0`` is skipped.
    Returns
    -------
    np.ndarray
        The 3-element weight vector ``[w1, w2, w3]`` (summing to 1, all
        non-negative) achieving the lowest loss found on the grid. 
    """

    if w0 is None:
       w0 = np.array([0, 0, 0], float)
    w0 = np.asarray(w0, float)
    best_loss = float("inf")
    best_w = np.array([1/3, 1/3, 1/3]) # Initial best, equal weights

    grid = np.arange(0.0, 1.0 + 1e-12, step)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                continue
            w = np.array([w1, w2, w3], dtype=float)
            r_safe_val = r_raw(r_safe, w)
            r_leak_val = r_raw(r_leak, w)
            loss = (r_safe_val - eps)**2 + (r_leak_val - lam)**2 + gamma * np.sum((w - w0)**2)
            if loss < best_loss:
                best_loss = loss
                best_w = w
    return best_w

def choose_targets_from_anchors(r_safe_list, r_leak_list):
    """Derive the calibration targets from collections of anchor risk vectors.
 
    Parameters
    ----------
    r_safe_list : iterable of sequence of float or None
        A collection of per-metric risk vectors, one per "safe" anchor
        sample (e.g. one entry per safe-cohort dataset evaluated).
    r_leak_list : iterable of sequence of float or None
        A collection of per-metric risk vectors, one per "leaky" anchor
        sample.
 
    Returns
    -------
    eps : float
        Target raw risk score for safe anchors, clipped to [0.02, 0.10].
    lam : float
        Target raw risk score for leaky anchors, clipped to [0.75, 0.90].
 
    """
    w0 = np.array([0.0, 0.0, 0.0])

    def R(r):
        # With w0 = 0, _or_like_raw_vec returns 0.0 by design (zero-sum weights)
        return r_raw(r, w0)

    R_safe_eq = float(np.median([R(r) for r in r_safe_list])) if len(r_safe_list) else 0.0
    R_leak_eq = float(np.median([R(r) for r in r_leak_list])) if len(r_leak_list) else 0.0

    eps = float(np.clip(R_safe_eq + 0.02, 0.02, 0.10))
    lam = float(np.clip(R_leak_eq + 0.10, 0.75, 0.90))
    return eps, lam

__all__ = ["_aggregate", 
           "clamp01", "r_raw",
           "calibrate_to_prism",
           "fit_weights_anchor",
           "choose_targets_from_anchors",]
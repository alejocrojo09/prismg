import numpy as np

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def r_raw(r_vec, w_vec=None):
    r = np.array([np.nan if v is None else float(v) for v in r_vec], dtype=float)
    keep = ~np.isnan(r)
    if not np.any(keep):
        return 0.0

    if w_vec is None:
        w = np.ones_like(r, dtype=float)
    else:
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
    denom = max(beta - alpha, eps)
    return 100.0 * clamp01((R_raw - alpha) / denom)

def _entropy_safe(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    p_pos = p[p > 0.0]
    if p_pos.size == 0:
        return 0.0
    return float(-(p_pos * np.log(p_pos)).sum())

def fit_weights_anchor(r_safe, r_leak, eps=0.05, lam=0.8, gamma=1e-3, step=0.01):
    w0 = np.array([0.0, 0.0, 0.0])
    best_loss = float("inf")
    best_w = np.array([1/3, 1/3, 1/3])

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
    w0 = np.array([0.0, 0.0, 0.0])

    def R(r):
        # With w0 = 0, _or_like_raw_vec returns 0.0 by design (zero-sum weights)
        return r_raw(r, w0)

    R_safe_eq = float(np.median([R(r) for r in r_safe_list])) if len(r_safe_list) else 0.0
    R_leak_eq = float(np.median([R(r) for r in r_leak_list])) if len(r_leak_list) else 0.0

    eps = float(np.clip(R_safe_eq + 0.02, 0.02, 0.10))
    lam = float(np.clip(R_leak_eq + 0.10, 0.75, 0.90))
    return eps, lam
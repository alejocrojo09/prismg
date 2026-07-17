from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from prismg.io.vcf_reader import load_vcf
 
VCFRowKey = Tuple[str, int, str, str]

# ----------------------------
# Ultra-rare variant mask (AF from cohort, NaN tolerant)
# ----------------------------
def make_ultra_rare_mask(G: np.ndarray, maf_thresh: float = 0.001, eps: float = 1e-9) -> np.ndarray:
    """Return a boolean mask of variants that are ultra-rare in the provided matrix.
 
    Parameters
    ----------
    G : np.ndarray
        Genotype dosage matrix.
        NaN entries are excluded from the frequency estimate.
    maf_thresh : float, default=0.001
        Variants with MAF strictly below this threshold are flagged as
        ultra-rare.
    eps : float, default=1e-9
        Small constant added to MAF before comparison to avoid
        floating-point equality issues at exactly zero.
 
    Returns
    -------
    np.ndarray
        Boolean mask.
    """
    G = np.asarray(G, dtype=float)
    sums = np.nansum(G, axis=0)  # sum of dosages
    nobs = np.sum(~np.isnan(G), axis=0)  # number of non-NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(nobs > 0, (sums / 2.0) / nobs, 0.0)
    maf = np.minimum(p, 1.0 - p)
    return (maf + eps) < maf_thresh

# ----------------------------
# Per-row genotype perturbation
# ----------------------------
def _flip_genotypes(dos_row: np.ndarray, flip_rate: float = 0.01, random_seed: int = 123, keep_ultra_rare_mask: Optional[np.ndarray] = None,) -> np.ndarray:
    """Perturb a single dosage row by randomly flipping genotypes by ±1.
 
    Parameters
    ----------
    dos_row : np.ndarray
        1D dosage array.
    flip_rate : float, default=0.01
        Per-SNP probability of being perturbed.
    random_seed : np.random.Generator, optional
        Random seed.
    keep_ultra_rare_mask : np.ndarray, optional
        Boolean mask of shape ``(m,)``. SNPs where this mask is ``True``
        are excluded from flipping.
 
    Returns
    -------
    np.ndarray
        Perturbed dosage row.
    """
    rng = np.random.default_rng(random_seed)
    out = dos_row.copy()
    m = out.shape[0]
    flips = rng.random(m) < flip_rate
    if keep_ultra_rare_mask is not None:
        flips = flips & (~keep_ultra_rare_mask)  # DO NOT touch ultra-rare variants
    # only flip where not NaN
    valid = ~np.isnan(out)
    idx = np.where(flips & valid)[0]
    if idx.size:
        step = rng.choice([-1.0, 1.0], size=idx.size)
        out[idx] = np.clip(out[idx] + step, 0.0, 2.0)
    return out

# ----------------------------
# Base leaky cohort: copy / near-dup / random mix
# ----------------------------
def generate_leaky_copycat(G_real: np.ndarray, n_samples: int = None, copy_frac: float = 0.60,
    neardup_frac: float = 0.30,
    random_frac: float = 0.10,
    flip_rate_neardup: float = 0.01,
    flip_rate_random: float = 0.05,
    keep_ultra_rare: bool = True,
    ultra_maf: float = 0.001,
    random_seed: int = 123,
) -> np.ndarray:
    """Build a leaky cohort by mixing exact copies, near-duplicates, and perturbed rows.

    Parameters
    ----------
    G_real : np.ndarray
        Real genotype dosage matrix.
    n_samples : int, optional
        Number of output rows. 
    copy_frac : float, default=0.60
        Fraction of output rows that are exact copies.
    neardup_frac : float, default=0.30
        Fraction of output rows that are near-duplicates (low flip rate).
    random_frac : float, default=0.10
        Fraction of output rows that are more heavily perturbed.
    flip_rate_neardup : float, default=0.01
        Per-SNP flip probability for near-duplicate rows.
    flip_rate_random : float, default=0.05
        Per-SNP flip probability for random rows.
    keep_ultra_rare : bool, default=True
        If ``True``, variants flagged as ultra-rare by
        ``make_ultra_rare_mask`` (MAF < ``ultra_maf``) are excluded from
        flipping in near-dup and random rows.
    ultra_maf : float, default=0.001
        MAF threshold for the ultra-rare mask. Only used when
        ``keep_ultra_rare=True``.
    random_seed : int, default=123
        Random seed
 
    Returns
    -------
    np.ndarray
        Leaky genotype dosage matrix.

    """
    assert abs(copy_frac + neardup_frac + random_frac - 1.0) < 1e-9
    rng = np.random.default_rng(random_seed)
    n_real, m = G_real.shape
    if n_samples is None:
        n_samples = n_real

    keep_rare_mask = make_ultra_rare_mask(G_real, maf_thresh=ultra_maf) if keep_ultra_rare else None
 
    n_copy = int(round(copy_frac * n_samples))
    n_near = int(round(neardup_frac * n_samples))
    n_rand = n_samples - n_copy - n_near
 
    base_idx = rng.choice(n_samples, size=n_samples, replace=True)
    Gb = G_real[base_idx, :]
 
    rows = []
    rows.extend(Gb[:n_copy, :])  # exact copies
 
    for i in range(n_copy, n_copy + n_near):
        rows.append(
            _flip_genotypes(Gb[i, :], flip_rate=flip_rate_neardup, random_seed = rng, keep_ultra_rare_mask=keep_rare_mask)
        )
    for i in range(n_copy + n_near, n_copy + n_near + n_rand):
        rows.append(
            _flip_genotypes(Gb[i, :], flip_rate=flip_rate_random, random_seed = rng, keep_ultra_rare_mask=keep_rare_mask)
        )
 
    return np.vstack(rows)

# ----------------------------
# Copycat-epsilon + kin-doping
# ----------------------------
def generate_leaky_kindoped(
    G_real: np.ndarray,
    n_samples: int,
    # base mix (copycat-epsilon)
    copy_frac: float = 0.50,
    neardup_frac: float = 0.40,
    random_frac: float = 0.10,
    flip_rate_neardup: float = 0.008,
    flip_rate_random: float = 0.03,
    # kin-doping controls (on the *output*)
    dup_pairs_frac: float = 0.20,
    sib_flip_rate: float = 0.008,
    keep_ultra_rare: bool = True,
    ultra_maf: float = 0.001,
    random_seed: int = 123,
) -> np.ndarray:
    """Build a leaky cohort via copycat-epsilon mixing followed by kin-doping.
 
    Runs in two steps:
 
    1. **Base cohort**: calls ``generate_leaky_copycat`` to produce a
       ``n_samples``-row matrix of exact copies, near-duplicates, and
       perturbed rows drawn from ``G_real``.
 
    2. **Kin-doping**: selects a random subset of output rows (fraction
       ``dup_pairs_frac``) and applies an additional low-rate flip via
       ``_flip_genotypes`` at ``sib_flip_rate``, simulating sibling-like
       near-duplicates within the output cohort itself.
 
    Parameters
    ----------
    G_real : np.ndarray
        Real genotype dosage matrix, shape ``(n_real, m)``.
    n_samples : int
        Number of output rows.
    copy_frac : float, default=0.50
        Fraction of base cohort rows that are exact copies.
    neardup_frac : float, default=0.40
        Fraction of base cohort rows that are near-duplicates.
    random_frac : float, default=0.10
        Fraction of base cohort rows that are more heavily perturbed.
        Must satisfy ``copy_frac + neardup_frac + random_frac == 1.0``.
    flip_rate_neardup : float, default=0.008
        Per-SNP flip rate for near-duplicate rows in the base cohort.
    flip_rate_random : float, default=0.03
        Per-SNP flip rate for random rows in the base cohort.
    dup_pairs_frac : float, default=0.20
        Fraction of the final output rows to kin-dope. Rows are
        selected uniformly at random without replacement.
    sib_flip_rate : float, default=0.008
        Per-SNP flip rate applied to kin-doped rows, simulating the
        level of divergence expected between siblings.
    keep_ultra_rare : bool, default=True
        Whether to protect ultra-rare variants from flipping in the
        base cohort step. Forwarded to ``generate_leaky_copycat``.
    ultra_maf : float, default=0.001
        MAF threshold for the ultra-rare mask. Forwarded to
        ``generate_leaky_copycat``.
    random_seed : int, default=123
        Random seed.
 
    Returns
    -------
    np.ndarray
        Leaky dosage matrix of shape ``(n_samples, m)``.
    """
    rng = np.random.default_rng(random_seed)

    # Step 1: base leaky cohort of size n_samples
    G_base = generate_leaky_copycat(
        G_real,
        n_samples=n_samples,
        copy_frac=copy_frac,
        neardup_frac=neardup_frac,
        random_frac=random_frac,
        flip_rate_neardup=flip_rate_neardup,
        flip_rate_random=flip_rate_random,
        keep_ultra_rare=keep_ultra_rare,
        ultra_maf=ultra_maf,
        random_seed=random_seed,
    )
 
    # Step 2: kin-doping
    k = int(round(dup_pairs_frac * n_samples))
    if k > 0:
        idx = rng.choice(n_samples, size=k, replace=False)
        for i in idx:
            G_base[i, :] = _flip_genotypes(G_base[i, :], flip_rate=sib_flip_rate, random_seed=rng)
 
    return G_base

# ----------------------------
# Dataset generator: real VCF -> (samples, meta, G) for the LEAKY cohort
# ----------------------------
def generate_leaky_dataset(
    real_vcf: str,
    n_samples: Optional[int] = None,
    random_seed: int = 123,
    # base mix (copycat-epsilon)
    copy_frac: float = 0.50,
    neardup_frac: float = 0.40,
    random_frac: float = 0.10,
    flip_rate_neardup: float = 0.008,
    flip_rate_random: float = 0.03,
    # kin-doping controls
    dup_pairs_frac: float = 0.20,
    sib_flip_rate: float = 0.008,
    keep_ultra_rare: bool = True,
    ultra_maf: float = 0.001,
    keep_pos=None,
    keep_full=None,
    sample_prefix: str = "LEAKY",
) -> Tuple[List[str], List[VCFRowKey], np.ndarray]:
    """Generate a leaky synthetic cohort from a real VCF panel.

    Parameters
    ----------
    real_vcf : str
        Path to a plain or gzip/bgzip-compressed VCF file. Passed
        directly to ``vcf_reader.load_vcf``.
    n_samples : int, optional
        Number of synthetic individuals to generate. Defaults to
        ``n_real``.
    random_seed : int, default=123
        Random seed.
    copy_frac : float, default=0.50
        Fraction of output rows that are exact copies of real
        individuals.
    neardup_frac : float, default=0.40
        Fraction of output rows that are near-duplicates.
    random_frac : float, default=0.10
        Fraction of output rows that are more heavily perturbed.
    flip_rate_neardup : float, default=0.008
        Per-SNP flip rate for near-duplicate rows.
    flip_rate_random : float, default=0.03
        Per-SNP flip rate for random rows.
    dup_pairs_frac : float, default=0.20
        Fraction of output rows to kin-dope.
    sib_flip_rate : float, default=0.008
        Per-SNP flip rate for kin-doped rows.
    keep_ultra_rare : bool, default=True
        Whether to protect ultra-rare variants from flipping.
    ultra_maf : float, default=0.001
        MAF threshold for the ultra-rare mask.
    keep_pos : set of (chrom, pos), optional
        Positional variant filter forwarded to ``load_vcf``. 
    keep_full : set of (chrom, pos, ref, alt), optional
        Exact variant filter forwarded to ``load_vcf``. Takes priority
        over ``keep_pos`` when both are provided.
    sample_prefix : str, default="LEAKY"
        Prefix for synthetic sample identifiers. 
 
    Returns
    -------
    samples : list of str
        Synthetic sample identifiers.
    meta : list of VCFRowKey
        ``(chrom, pos, ref, alt)`` tuples from the real VCF, in panel
        order. 
    G : np.ndarray
        Leaky genotype dosage matrix.
    """
    real_samples, real_meta, G_real = load_vcf(real_vcf, keep_full=keep_full, keep_pos=keep_pos)

    # 3) Build the LEAKY cohort from the train split
    G_leaky = generate_leaky_kindoped(
        G_real,
        n_samples=n_samples,
        copy_frac=copy_frac,
        neardup_frac=neardup_frac,
        random_frac=random_frac,
        flip_rate_neardup=flip_rate_neardup,
        flip_rate_random=flip_rate_random,
        dup_pairs_frac=dup_pairs_frac,
        sib_flip_rate=sib_flip_rate,
        keep_ultra_rare=keep_ultra_rare,
        ultra_maf=ultra_maf,
        random_seed=random_seed,
    )
 
    # 4) Build sample names; meta is reused as-is from REAL (same panel order)
    leaky_samples = [f"{sample_prefix}_{i:04d}" for i in range(n_samples)]
 
    return leaky_samples, real_meta, G_leaky

__all__ = ["make_ultra_rare_mask", "_flip_genotypes",
           "generate_leaky_copycat", "generate_leaky_kindoped",
           "generate_leaky_dataset",]

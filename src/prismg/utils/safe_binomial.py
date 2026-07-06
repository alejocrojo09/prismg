from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import gzip

from prismg.io.vcf_reader import load_vcf
from prismg.utils.grm import _af_from_ref
 
VCFRowKey = Tuple[str, int, str, str]
 

def af_matched_null_safe(G_tr: np.ndarray, n_samples: int, random_seed: int = 123, floor_maf: Optional[float] = None,) -> np.ndarray:
    """Sample a synthetic dosage matrix with a binomial distribution using AFs from a reference panel.

    Parameters
    ----------
    G_tr : np.ndarray
        Reference genotype dosage matrix.
    n_samples : int
        Number of synthetic individuals to generate.
    random_seed : int, default=123
        Random seed.
    floor_maf : float, optional
        If provided, pushes ultra-rare variants away from the allele
        frequency boundaries: any SNP with MAF below ``floor_maf`` has
        its frequency moved to ``floor_maf`` (for rare ALT alleles) or
        ``1 - floor_maf`` (for rare REF alleles), while preserving which
        allele is major. Use this to suppress ultra-rare variants in the
        safe cohort without removing them from the panel entirely.
 
    Returns
    -------
    np.ndarray
        Synthetic genotype dosage matrix.
    """
    rng = np.random.default_rng(random_seed)
    p = _af_from_ref(G_tr)  # AF per SNP from REAL train
    if floor_maf is not None:
        maf = np.minimum(p, 1.0 - p)
        low = maf < floor_maf
        # move probability away from boundary while keeping the major allele
        p[low] = np.where(p[low] < 0.5, floor_maf, 1.0 - floor_maf)
 
    m = p.size
    H1 = rng.binomial(1, p, size=(n_samples, m))
    H2 = rng.binomial(1, p, size=(n_samples, m))
    return (H1 + H2).astype(float)  # 0/1/2 dosages
 

def generate_safe_dataset(real_vcf: str, n_safe: int, random_seed: int = 123, floor_maf: Optional[float] = None, keep_pos=None, keep_full=None, sample_prefix: str = "SAFE",) -> Tuple[List[str], List[VCFRowKey], np.ndarray]:
    """Generate an AF-matched safe synthetic cohort from a real VCF panel.
 
    Parameters
    ----------
    real_vcf : str
        Path to a plain or gzip/bgzip-compressed VCF file. 
    n_safe : int
        Number of synthetic individuals to generate.
    random_seed : int, default=123
        Random seed.
    floor_maf : float, optional
        Ultra-rare variant MAF floor. If ``None``, allele frequencies are
        used as estimated without adjustment.
    keep_pos : set of (chrom, pos), optional
        Positional variant filter forwarded to ``load_vcf``. Only
        variants whose ``(chrom, pos)`` is in this set are loaded.
    keep_full : set of (chrom, pos, ref, alt), optional
        Exact variant filter forwarded to ``load_vcf``.
    sample_prefix : str, default="SAFE"
        Prefix for synthetic sample identifiers. 
 
    Returns
    -------
    samples : list of str
        Synthetic sample identifiers.
    meta : list of VCFRowKey
        ``(chrom, pos, ref, alt)`` tuples from the real VCF, in panel
        order. Identical to what ``load_vcf`` would return for the same
        file and filters.
    G : np.ndarray
        Synthetic genotype dosage matrix.
    """
    real_samples, real_meta, G_real = load_vcf(real_vcf, keep_full=keep_full, keep_pos=keep_pos)
    G_safe = af_matched_null_safe(G_real, n_samples=n_safe, random_seed=random_seed, floor_maf=floor_maf)
    safe_samples = [f"{sample_prefix}_{i:04d}" for i in range(n_safe)]
    return safe_samples, real_meta, G_safe
 
__all__ = ["af_matched_null_safe",
           "generate_safe_dataset",]
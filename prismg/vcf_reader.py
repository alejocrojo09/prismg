"""
Lightweight VCF readers for PRISM-G — keep it simple and predictable.

This version mirrors the user's working logic and adds only two safe extras:
- transparent support for .vcf.gz / .vcf.bgz
- optional filtering by either FULL key (chrom,pos,ref,alt) *or* POS-only key (chrom,pos)

API
---
load_vcf(path, keep_full=None, keep_pos=None) -> (samples, meta, G)
  samples : list[str]
  meta    : list[(chrom,pos,ref,alt)] in column order of G
  G       : np.ndarray [n_samples, n_snps] with {0,1,2} and np.nan

Notes
-----
- Multi-allelic ALT is truncated to the first allele (ALT[0]) to match the prototype behavior.
- No chromosome normalization is performed; values are used exactly as they appear in the VCF.
"""
from __future__ import annotations

import gzip
from typing import List, Optional, Tuple, Set

import numpy as np
import pandas as pd

VCFRowKey = Tuple[str, int, str, str]
PosKey = Tuple[str, int]

# ==================== I/O helpers ====================

def _open_text(path: str):
    if path.endswith(".gz") or path.endswith(".bgz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


# ==================== helpers ====================

def collect_variants(path: str) -> Set[VCFRowKey]:
    """Collect (chrom, pos, ref, alt) for rows with at least 10 columns."""
    ids: List[VCFRowKey] = []
    with _open_text(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue
            chrom, pos, _id, ref, alt = parts[0], int(parts[1]), parts[2], parts[3], parts[4]
            ids.append((chrom, pos, ref, alt.split(",")[0]))
    return set(ids)


# ==================== core loader ====================

def load_vcf(
    path: str,
    keep_full: Optional[Set[VCFRowKey]] = None,
    keep_pos: Optional[Set[PosKey]] = None,
):
    """
    Load a (plain or gz) VCF to dosage matrix in {0,1,2} (./. → NaN).

    Filtering behavior:
      - If keep_full is provided, keep variant only if (chrom,pos,ref,alt) ∈ keep_full
      - Else if keep_pos is provided, keep if (chrom,pos) ∈ keep_pos
      - Else keep all variants
    """
    samples: List[str] = []
    meta: List[VCFRowKey] = []
    cols: List[np.ndarray] = []

    with _open_text(path) as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.rstrip("\n").split("\t")
                samples = header[9:]
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue
            chrom, pos, _id, ref, alt = parts[0], int(parts[1]), parts[2], parts[3], parts[4]
            alt0 = alt.split(",")[0]
            key_full: VCFRowKey = (chrom, pos, ref, alt0)
            key_pos: PosKey = (chrom, pos)

            if keep_full is not None and key_full not in keep_full:
                continue
            if keep_full is None and keep_pos is not None and key_pos not in keep_pos:
                continue

            genos = parts[9:]
            dos = []
            for g in genos:
                gt = g.split(":", 1)[0]
                if gt in ("./.", ".|."):
                    dos.append(np.nan)
                else:
                    a = gt.replace("|", "/").split("/")
                    try:
                        dos.append(sum(int(x) for x in a))
                    except Exception:
                        dos.append(np.nan)
            meta.append(key_full)
            cols.append(np.asarray(dos, dtype=float))

    G = np.vstack(cols).T if cols else np.empty((len(samples), 0))
    return samples, meta, G


# ==================== legend utilities (optional) ====================

def load_snp_legend_pos_keys(path: str) -> Set[PosKey]:
    """Load legend with an 'id' column like '15:12345_A_G' or 'chr15:12345_A_G' → {(chrom,pos)}.
    Chromosome strings are used *as-is* (no 'chr' normalization) to stay predictable with the VCF.
    """
    df = pd.read_csv(path, sep=r"\s+|\t|,", engine="python")
    col = "id" if "id" in df.columns else df.columns[0]

    def _key(val: str) -> PosKey:
        s = str(val)
        chrom, rest = s.split(":", 1)
        pos = int(rest.split("_", 1)[0])
        return chrom, pos

    return set(df[col].apply(_key).tolist())


__all__ = [
    "collect_variants",
    "load_vcf",
    "load_snp_legend_pos_keys",
]

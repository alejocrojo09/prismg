from __future__ import annotations
from typing import List, Optional, Tuple, Set

import gzip
import numpy as np
import pandas as pd

VCFRowKey = Tuple[str, int, str, str]
PosKey = Tuple[str, int]

def _open_text(path: str):
    if path.endswith(".gz") or path.endswith(".bgz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")

def collect_variants(path: str) -> Set[VCFRowKey]:
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

def load_vcf(path: str, keep_full: Optional[Set[VCFRowKey]] = None, keep_pos: Optional[Set[PosKey]] = None):
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

def load_snp_legend_pos_keys(path: str) -> Set[PosKey]:
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

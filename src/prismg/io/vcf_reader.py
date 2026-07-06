from __future__ import annotations
from typing import List, Optional, Tuple, Set

import gzip
import numpy as np
import pandas as pd

VCFRowKey = Tuple[str, int, str, str]
PosKey = Tuple[str, int]

def _open_text(path: str):
    """Open a plain or gzip/bgzip-compressed text file for reading.
 
    Parameters
    ----------
    path : str
        File path. 
    Returns
    -------
    file-like object
        An open text-mode file handle. 
    """
    if path.endswith(".gz") or path.endswith(".bgz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")

def collect_variants(path: str) -> Set[VCFRowKey]:
    """Collect the set of all variant keys present in a VCF file.
 
    Parameters
    ----------
    path : str
        Path to a plain or gzip/bgzip-compressed VCF file.
 
    Returns
    -------
    set of VCFRowKey
        Set of ``(chrom, pos, ref, alt)`` tuples for every variant
        line.
    """
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
    """Load a VCF file into a dosage matrix with optional variant filtering.

    Filtering is controlled by two mutually exclusive keyword arguments:
 
    - If ``keep_full`` is provided, only variants whose
      ``(chrom, pos, ref, alt)`` key is in that set are loaded.
    - If ``keep_full`` is ``None`` and ``keep_pos`` is provided, only
      variants whose ``(chrom, pos)`` is in that set are loaded.
    - If both are ``None``, all variants are loaded.
 
    Parameters
    ----------
    path : str
        Path to a plain or gzip/bgzip-compressed VCF file.
    keep_full : set of VCFRowKey, optional
        Exact ``(chrom, pos, ref, alt)`` filter. Takes priority over
        ``keep_pos`` when both are provided (``keep_pos`` is ignored).
    keep_pos : set of PosKey, optional
        Positional ``(chrom, pos)`` filter. Only used when ``keep_full``
        is ``None``.
 
    Returns
    -------
    samples : list of str
        Sample identifiers.
    meta : list of VCFRowKey
        ``(chrom, pos, ref, alt)`` tuples for each loaded variant, in
        file order. 
    G : np.ndarray
        Genotype dosage matrix.
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

def load_snp_legend_pos_keys(path: str) -> Set[PosKey]:
    """Load a SNP legend file and return a set of positional variant keys.

    Parameters
    ----------
    path : str
        Path to the SNP legend file.
 
    Returns
    -------
    set of PosKey
        Set of ``(chrom, pos)`` tuples, one per row of the legend file.
    """
    df = pd.read_csv(path, sep=r"\s+|\t|,", engine="python")
    col = "id" if "id" in df.columns else df.columns[0]

    def _key(val: str) -> PosKey:
        s = str(val)
        chrom, rest = s.split(":", 1)
        pos = int(rest.split("_", 1)[0])
        return chrom, pos

    return set(df[col].apply(_key).tolist())

def write_vcf(out_path: str, samples: List[str], meta: List[VCFRowKey], G: np.ndarray, gzip_output: bool = True,) -> None:
    """Write a dosage matrix back to a VCF file.
 
    Parameters
    ----------
    out_path : str
        Output file path. 
    samples : list of str
        Sample identifiers to write as column headers, in order.
    meta : list of VCFRowKey
        ``(chrom, pos, ref, alt)`` tuples in panel order.
    G : np.ndarray
        Genotype dosage matrix.
    gzip_output : bool, default=True
        If ``True``, write gzip-compressed output. 
 
    """
    sep = "|" 
    n, m = G.shape
    assert m == len(meta), "G columns must match panel length"
 
    opener = (lambda p: gzip.open(p, "wt", encoding="utf-8")) if gzip_output else (
        lambda p: open(p, "w", encoding="utf-8")
    )
 
    with opener(out_path) as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        header = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + samples
        f.write("\t".join(header) + "\n")
        for j, (chrom, pos, ref, alt) in enumerate(meta):
            col = G[:, j]
            gts = []
            for d in col:
                d = int(round(d))
                if d <= 0:
                    gts.append(f"0{sep}0")
                elif d == 1:
                    gts.append(f"0{sep}1")
                else:
                    gts.append(f"1{sep}1")
            row = [str(chrom), str(int(pos)), ".", str(ref), str(alt), ".", "PASS", ".", "GT"] + gts
            f.write("\t".join(row) + "\n")

__all__ = [
    "collect_variants",
    "load_vcf",
    "load_snp_legend_pos_keys",
    "write_vcf",
]

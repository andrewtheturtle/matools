"""
matools.io
----------
Helpers for reading and writing common cancer-genomics file formats.

Supported formats
~~~~~~~~~~~~~~~~~
* MAF  (Mutation Annotation Format) – tab-delimited, ``#`` comment lines ignored
* VCF  (Variant Call Format)         – ``#`` header lines preserved separately
* BED  (Browser Extensible Data)     – 3-column minimum, optional extra columns
* TSV  (generic tab-separated)       – first non-comment line treated as header

All readers return :class:`pandas.DataFrame` objects; writers accept DataFrames.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import IO, Optional, Union

import pandas as pd

PathLike = Union[str, os.PathLike]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open(path: PathLike, mode: str = "rt") -> IO:
    """Open a plain or gzip-compressed file transparently."""
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, mode)
    return open(p, mode)


def _skip_comment_lines(path: PathLike, comment: str = "#") -> int:
    """Return the number of leading comment/blank lines to skip."""
    skip = 0
    with _open(path) as fh:
        for line in fh:
            if line.startswith(comment) or line.strip() == "":
                skip += 1
            else:
                break
    return skip


# ---------------------------------------------------------------------------
# MAF
# ---------------------------------------------------------------------------

# Columns that are always expected in a valid MAF file.
MAF_REQUIRED_COLS = [
    "Hugo_Symbol",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Variant_Classification",
    "Variant_Type",
    "Reference_Allele",
    "Tumor_Seq_Allele1",
    "Tumor_Seq_Allele2",
    "Tumor_Sample_Barcode",
]


def read_maf(
    path: PathLike,
    low_memory: bool = False,
    comment: str = "#",
) -> pd.DataFrame:
    """Read a MAF file into a DataFrame.

    Parameters
    ----------
    path:
        Path to the ``.maf`` or ``.maf.gz`` file.
    low_memory:
        Passed directly to :func:`pandas.read_csv`.  Set to ``True`` for very
        large files when memory is constrained (may infer dtypes per chunk).
    comment:
        Lines that start with this character are treated as comments and
        skipped.  The default ``"#"`` is standard for MAF files.

    Returns
    -------
    pandas.DataFrame
        One row per mutation.  Column names are taken from the first
        non-comment line of the file.
    """
    skiprows = _skip_comment_lines(path, comment=comment)
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=skiprows,
        low_memory=low_memory,
        dtype=str,
    )
    df.columns = df.columns.str.strip()
    return df


def write_maf(df: pd.DataFrame, path: PathLike, comment_header: str = "") -> None:
    """Write a DataFrame to a MAF-formatted file.

    Parameters
    ----------
    df:
        DataFrame to write.  Should contain at minimum the columns in
        :data:`MAF_REQUIRED_COLS`.
    path:
        Destination file path.  Use a ``.gz`` suffix to compress on the fly.
    comment_header:
        Optional free-text block prepended before the column header (each line
        automatically prefixed with ``#version ``).
    """
    with _open(path, "wt") as fh:
        if comment_header:
            for line in comment_header.splitlines():
                fh.write(f"#version {line}\n")
        df.to_csv(fh, sep="\t", index=False)


# ---------------------------------------------------------------------------
# VCF
# ---------------------------------------------------------------------------

def read_vcf(path: PathLike) -> tuple[list[str], pd.DataFrame]:
    """Read a VCF file.

    Returns
    -------
    header_lines : list[str]
        All ``##`` meta-information lines.
    variants : pandas.DataFrame
        The variant records.  Column names are derived from the ``#CHROM``
        header line (``#CHROM`` is renamed to ``CHROM``).
    """
    header_lines: list[str] = []
    col_line: Optional[str] = None

    with _open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("##"):
                header_lines.append(line)
            elif line.startswith("#CHROM"):
                col_line = line.lstrip("#")
                break

    if col_line is None:
        raise ValueError(f"No '#CHROM' header line found in {path}")

    columns = col_line.split("\t")
    skip = len(header_lines) + 1  # +1 for the #CHROM line

    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=columns,
        skiprows=skip,
        dtype=str,
        low_memory=False,
    )
    return header_lines, df


# ---------------------------------------------------------------------------
# BED
# ---------------------------------------------------------------------------

BED_BASE_COLS = ["chrom", "chromStart", "chromEnd"]


def read_bed(path: PathLike, extra_cols: Optional[list[str]] = None) -> pd.DataFrame:
    """Read a BED file into a DataFrame.

    Parameters
    ----------
    path:
        Path to the ``.bed`` or ``.bed.gz`` file.
    extra_cols:
        Names for any columns beyond the mandatory three.  If ``None`` the
        extra columns are labelled ``col4``, ``col5``, …

    Returns
    -------
    pandas.DataFrame
    """
    skiprows = _skip_comment_lines(path)

    # Peek at the first data line to count columns.
    with _open(path) as fh:
        for line in fh:
            if not line.startswith("#") and line.strip():
                n_cols = len(line.split("\t"))
                break
        else:
            n_cols = 3

    if extra_cols is None:
        extra_cols = [f"col{i + 1}" for i in range(3, n_cols)]

    names = BED_BASE_COLS + extra_cols

    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=skiprows,
        header=None,
        names=names,
        dtype=str,
        low_memory=False,
    )
    df["chromStart"] = df["chromStart"].astype(int)
    df["chromEnd"] = df["chromEnd"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Generic TSV
# ---------------------------------------------------------------------------

def read_tsv(path: PathLike, comment: str = "#", **kwargs) -> pd.DataFrame:
    """Read a generic tab-separated file, skipping comment lines.

    Parameters
    ----------
    path:
        Path to the file.
    comment:
        Lines beginning with this character are skipped.
    **kwargs:
        Additional keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
    """
    skiprows = _skip_comment_lines(path, comment=comment)
    return pd.read_csv(path, sep="\t", skiprows=skiprows, low_memory=False, **kwargs)


def write_tsv(df: pd.DataFrame, path: PathLike, index: bool = False) -> None:
    """Write a DataFrame as a tab-separated file.

    Parameters
    ----------
    df:
        DataFrame to write.
    path:
        Destination path.
    index:
        Whether to include the DataFrame index.
    """
    df.to_csv(path, sep="\t", index=index)

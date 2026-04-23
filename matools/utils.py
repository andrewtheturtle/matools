"""
matools.utils
-------------
General-purpose helpers for cancer genomics analyses.

Key functions
~~~~~~~~~~~~~
* :func:`parse_genomic_region`  – parse ``"chr1:1000-2000"`` strings
* :func:`overlap_regions`       – find overlapping BED-style intervals
* :func:`complement_strand`     – reverse-complement a DNA sequence
* :func:`gc_content`            – compute GC fraction for a sequence
* :func:`load_gene_list`        – read a single-column gene list from a text file
* :func:`chunks`                – split an iterable into fixed-size batches
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator, Iterable, TypeVar

import pandas as pd

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Genomic-region utilities
# ---------------------------------------------------------------------------

_REGION_RE = re.compile(
    r"^(?P<chrom>[^\s:]+):(?P<start>\d[\d,]*)-(?P<end>\d[\d,]*)$"
)


def parse_genomic_region(region: str) -> dict[str, str | int]:
    """Parse a UCSC-style genomic region string.

    Parameters
    ----------
    region:
        String of the form ``"chr1:1,000-2,000"`` or ``"1:1000-2000"``.
        Commas in coordinates are accepted and stripped.

    Returns
    -------
    dict with keys ``"chrom"`` (str), ``"start"`` (int), ``"end"`` (int).

    Raises
    ------
    ValueError
        If *region* does not match the expected format.
    """
    m = _REGION_RE.match(region.strip())
    if m is None:
        raise ValueError(
            f"Cannot parse genomic region '{region}'.  "
            "Expected format: 'chrom:start-end' (e.g. 'chr7:140453136-140453137')."
        )
    return {
        "chrom": m.group("chrom"),
        "start": int(m.group("start").replace(",", "")),
        "end": int(m.group("end").replace(",", "")),
    }


def overlap_regions(
    query: pd.DataFrame,
    subject: pd.DataFrame,
    chrom_col: str = "chrom",
    start_col: str = "chromStart",
    end_col: str = "chromEnd",
) -> pd.DataFrame:
    """Return rows from *query* that overlap at least one interval in *subject*.

    This is a simple O(n·m) implementation suitable for small to medium-sized
    interval sets.  For very large datasets consider using pybedtools or
    PyRanges instead.

    Parameters
    ----------
    query:
        BED-like DataFrame to filter.
    subject:
        BED-like DataFrame defining the intervals of interest.
    chrom_col, start_col, end_col:
        Column names used for chromosome, start, and end in both DataFrames.

    Returns
    -------
    pandas.DataFrame
        Subset of *query* rows that overlap any interval in *subject*.
    """
    hits = []
    for chrom, sub_grp in subject.groupby(chrom_col):
        q_chrom = query[query[chrom_col] == chrom]
        if q_chrom.empty:
            continue
        for _, s_row in sub_grp.iterrows():
            s_start, s_end = s_row[start_col], s_row[end_col]
            mask = (q_chrom[end_col] > s_start) & (q_chrom[start_col] < s_end)
            hits.append(q_chrom[mask])
    if not hits:
        return query.iloc[:0].copy()
    return pd.concat(hits).drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------

_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def complement_strand(seq: str) -> str:
    """Return the reverse complement of a DNA sequence.

    Parameters
    ----------
    seq:
        DNA string (A, C, G, T, N accepted; case preserved).

    Returns
    -------
    str
        Reverse-complemented sequence.
    """
    return seq.translate(_COMPLEMENT)[::-1]


def gc_content(seq: str) -> float:
    """Compute the GC fraction of a DNA sequence.

    Parameters
    ----------
    seq:
        DNA string.  Non-ACGT characters (including N) are excluded from the
        denominator.

    Returns
    -------
    float
        GC fraction in [0, 1], or ``float('nan')`` if the sequence contains no
        ACGT characters.
    """
    seq_upper = seq.upper()
    acgt = sum(seq_upper.count(b) for b in "ACGT")
    if acgt == 0:
        return float("nan")
    gc = seq_upper.count("G") + seq_upper.count("C")
    return gc / acgt


# ---------------------------------------------------------------------------
# Gene-list helpers
# ---------------------------------------------------------------------------


def load_gene_list(path: str | Path, comment: str = "#") -> list[str]:
    """Read a plain-text gene list (one gene symbol per line).

    Parameters
    ----------
    path:
        Path to the text file.
    comment:
        Lines beginning with this character are treated as comments and
        skipped.

    Returns
    -------
    list[str]
        Stripped, non-empty gene symbols in file order.
    """
    genes = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith(comment):
                genes.append(line)
    return genes


def filter_genes(
    df: pd.DataFrame,
    genes: Iterable[str],
    gene_col: str = "Hugo_Symbol",
) -> pd.DataFrame:
    """Filter a DataFrame to rows whose gene is in *genes*.

    Parameters
    ----------
    df:
        DataFrame with a gene column.
    genes:
        Iterable of gene symbols to keep.
    gene_col:
        Name of the gene-symbol column.

    Returns
    -------
    pandas.DataFrame
    """
    return df[df[gene_col].isin(set(genes))].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------


def chunks(iterable: Iterable[T], size: int) -> Generator[list[T], None, None]:
    """Split an iterable into successive fixed-size chunks.

    The final chunk may be smaller than *size*.

    Parameters
    ----------
    iterable:
        Any iterable.
    size:
        Maximum number of elements per chunk.

    Yields
    ------
    list
        Successive sub-lists of at most *size* elements.

    Examples
    --------
    >>> list(chunks(range(10), 3))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    if size <= 0:
        raise ValueError(f"size must be a positive integer, got {size}")
    batch: list[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def flatten(nested: Iterable[Iterable[T]]) -> list[T]:
    """Flatten one level of nesting.

    Parameters
    ----------
    nested:
        An iterable of iterables.

    Returns
    -------
    list
        All elements from the inner iterables in order.

    Examples
    --------
    >>> flatten([[1, 2], [3], [4, 5, 6]])
    [1, 2, 3, 4, 5, 6]
    """
    return [item for sub in nested for item in sub]

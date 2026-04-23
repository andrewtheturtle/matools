"""
matools.variants
----------------
Helpers for filtering, annotating, and summarising somatic variants stored in
:class:`pandas.DataFrame` objects (typically loaded from a MAF file via
:mod:`matools.io`).

Key functions
~~~~~~~~~~~~~
* :func:`filter_by_classification` – keep/drop specific Variant_Classification values
* :func:`filter_by_type`           – keep/drop Variant_Type values (SNP, INS, DEL, …)
* :func:`filter_by_vaf`            – filter by tumour variant-allele fraction
* :func:`filter_nonsynonymous`     – keep only protein-altering mutations
* :func:`add_vaf`                  – compute VAF from allele-count columns
* :func:`summarise_variants`       – per-sample or per-gene mutation counts
* :func:`most_frequent_genes`      – rank genes by mutation frequency across samples
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Classification / type constants
# ---------------------------------------------------------------------------

#: Variant_Classification values considered protein-altering (non-synonymous).
NONSYNONYMOUS_CLASSIFICATIONS = frozenset(
    [
        "Missense_Mutation",
        "Nonsense_Mutation",
        "Frame_Shift_Del",
        "Frame_Shift_Ins",
        "In_Frame_Del",
        "In_Frame_Ins",
        "Splice_Site",
        "Translation_Start_Site",
        "Nonstop_Mutation",
    ]
)

#: Variant_Classification values that represent silent / synonymous mutations.
SYNONYMOUS_CLASSIFICATIONS = frozenset(
    [
        "Silent",
        "3'UTR",
        "5'UTR",
        "3'Flank",
        "5'Flank",
        "Intron",
        "IGR",
        "RNA",
        "lincRNA",
    ]
)


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------


def filter_by_classification(
    df: pd.DataFrame,
    keep: Optional[Iterable[str]] = None,
    drop: Optional[Iterable[str]] = None,
    col: str = "Variant_Classification",
) -> pd.DataFrame:
    """Filter variants by Variant_Classification.

    Parameters
    ----------
    df:
        MAF-like DataFrame.
    keep:
        If provided, only rows whose *col* value is in *keep* are retained.
    drop:
        If provided, rows whose *col* value is in *drop* are removed.
        Applied *after* ``keep``.
    col:
        Column name that holds the classification.  Defaults to the standard
        MAF column name.

    Returns
    -------
    pandas.DataFrame
        Filtered copy of *df*.
    """
    result = df.copy()
    if keep is not None:
        result = result[result[col].isin(set(keep))]
    if drop is not None:
        result = result[~result[col].isin(set(drop))]
    return result.reset_index(drop=True)


def filter_by_type(
    df: pd.DataFrame,
    keep: Optional[Iterable[str]] = None,
    drop: Optional[Iterable[str]] = None,
    col: str = "Variant_Type",
) -> pd.DataFrame:
    """Filter variants by Variant_Type (SNP, DNP, TNP, ONP, INS, DEL, …).

    Parameters mirror :func:`filter_by_classification`.
    """
    return filter_by_classification(df, keep=keep, drop=drop, col=col)


def filter_nonsynonymous(df: pd.DataFrame, col: str = "Variant_Classification") -> pd.DataFrame:
    """Retain only non-synonymous (protein-altering) mutations.

    Uses :data:`NONSYNONYMOUS_CLASSIFICATIONS` as the allowed set.
    """
    return filter_by_classification(df, keep=NONSYNONYMOUS_CLASSIFICATIONS, col=col)


def filter_by_vaf(
    df: pd.DataFrame,
    min_vaf: float = 0.0,
    max_vaf: float = 1.0,
    vaf_col: str = "VAF",
) -> pd.DataFrame:
    """Filter variants by tumour variant-allele fraction.

    Parameters
    ----------
    df:
        DataFrame that must contain *vaf_col* (add it first with
        :func:`add_vaf` if necessary).
    min_vaf:
        Minimum VAF (inclusive).
    max_vaf:
        Maximum VAF (inclusive).
    vaf_col:
        Column name for the VAF values.

    Returns
    -------
    pandas.DataFrame
    """
    if vaf_col not in df.columns:
        raise KeyError(
            f"Column '{vaf_col}' not found.  Run add_vaf() first or specify vaf_col."
        )
    mask = df[vaf_col].between(min_vaf, max_vaf)
    return df[mask].reset_index(drop=True)


def filter_by_sample(
    df: pd.DataFrame,
    samples: Iterable[str],
    col: str = "Tumor_Sample_Barcode",
) -> pd.DataFrame:
    """Retain only variants belonging to the specified sample(s).

    Parameters
    ----------
    df:
        MAF-like DataFrame.
    samples:
        Sample barcode(s) to keep.
    col:
        Column used to identify samples.

    Returns
    -------
    pandas.DataFrame
    """
    return df[df[col].isin(set(samples))].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------


def add_vaf(
    df: pd.DataFrame,
    ref_col: str = "t_ref_count",
    alt_col: str = "t_alt_count",
    vaf_col: str = "VAF",
) -> pd.DataFrame:
    """Compute the tumour variant-allele fraction and add it as a new column.

    Parameters
    ----------
    df:
        MAF-like DataFrame containing tumour reference and alternate read-count
        columns.
    ref_col:
        Column name for tumour reference read counts.
    alt_col:
        Column name for tumour alternate read counts.
    vaf_col:
        Name of the output column.

    Returns
    -------
    pandas.DataFrame
        Copy of *df* with *vaf_col* appended.
    """
    df = df.copy()
    ref = pd.to_numeric(df[ref_col], errors="coerce")
    alt = pd.to_numeric(df[alt_col], errors="coerce")
    total = ref + alt
    df[vaf_col] = alt / total.where(total > 0)
    return df


def add_mutation_class(
    df: pd.DataFrame,
    col: str = "Variant_Classification",
    out_col: str = "Mutation_Class",
) -> pd.DataFrame:
    """Add a simplified mutation class column (``'nonsynonymous'``, ``'synonymous'``,
    or ``'other'``).

    Parameters
    ----------
    df:
        MAF-like DataFrame.
    col:
        Column containing the detailed Variant_Classification values.
    out_col:
        Name of the new column.

    Returns
    -------
    pandas.DataFrame
    """
    df = df.copy()

    def _classify(val: str) -> str:
        if val in NONSYNONYMOUS_CLASSIFICATIONS:
            return "nonsynonymous"
        if val in SYNONYMOUS_CLASSIFICATIONS:
            return "synonymous"
        return "other"

    df[out_col] = df[col].apply(_classify)
    return df


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def summarise_variants(
    df: pd.DataFrame,
    by: str = "Tumor_Sample_Barcode",
    classification_col: str = "Variant_Classification",
) -> pd.DataFrame:
    """Count mutations per sample (or per gene) broken down by classification.

    Parameters
    ----------
    df:
        MAF-like DataFrame.
    by:
        Column to group by.  Use ``"Tumor_Sample_Barcode"`` for per-sample
        counts or ``"Hugo_Symbol"`` for per-gene counts.
    classification_col:
        Column whose values define the columns in the output pivot table.

    Returns
    -------
    pandas.DataFrame
        A pivot table with ``by`` as the index and one column per unique
        classification value, plus a ``Total`` column.
    """
    pivot = (
        df.groupby([by, classification_col])
        .size()
        .unstack(fill_value=0)
    )
    pivot["Total"] = pivot.sum(axis=1)
    return pivot.reset_index()


def most_frequent_genes(
    df: pd.DataFrame,
    n: int = 20,
    gene_col: str = "Hugo_Symbol",
    sample_col: str = "Tumor_Sample_Barcode",
    count_method: str = "samples",
) -> pd.DataFrame:
    """Rank genes by mutation frequency.

    Parameters
    ----------
    df:
        MAF-like DataFrame (typically pre-filtered to non-synonymous mutations).
    n:
        Return the top *n* genes.
    gene_col:
        Column containing gene symbols.
    sample_col:
        Column containing sample identifiers.
    count_method:
        ``"samples"`` – number of unique samples mutated (default);
        ``"mutations"`` – total number of mutations.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``[gene_col, "count"]`` sorted descending.
    """
    if count_method == "samples":
        counts = (
            df.groupby(gene_col)[sample_col]
            .nunique()
            .rename("count")
        )
    elif count_method == "mutations":
        counts = df[gene_col].value_counts().rename("count")
    else:
        raise ValueError(f"count_method must be 'samples' or 'mutations', got '{count_method}'")

    return (
        counts.nlargest(n)
        .reset_index()
        .rename(columns={"index": gene_col})
    )


def mutation_spectrum(
    df: pd.DataFrame,
    ref_col: str = "Reference_Allele",
    alt_col: str = "Tumor_Seq_Allele2",
) -> pd.Series:
    """Compute the single-nucleotide substitution spectrum for SNPs.

    Returns a :class:`pandas.Series` with counts for each of the six canonical
    substitution types (C>A, C>G, C>T, T>A, T>C, T>G).

    Parameters
    ----------
    df:
        MAF-like DataFrame filtered to SNPs only.
    ref_col:
        Column containing the reference allele.
    alt_col:
        Column containing the tumour alternate allele.

    Returns
    -------
    pandas.Series
        Substitution type counts, indexed by strings like ``'C>T'``.
    """
    snp_mask = df[ref_col].str.len() == 1
    pairs = df.loc[snp_mask, ref_col].str.upper() + ">" + df.loc[snp_mask, alt_col].str.upper()

    # Collapse to pyrimidine context (C or T as reference)
    complement = str.maketrans("ACGT", "TGCA")

    def _normalise(pair: str) -> str:
        ref, alt = pair[0], pair[2]
        if ref in ("C", "T"):
            return f"{ref}>{alt}"
        ref_c = ref.translate(complement)
        alt_c = alt.translate(complement)
        return f"{ref_c}>{alt_c}"

    normalised = pairs.dropna().apply(_normalise)
    canonical = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
    spectrum = normalised.value_counts().reindex(canonical, fill_value=0)
    return spectrum

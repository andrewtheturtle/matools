"""
matools.cohort
--------------
Cohort-level helpers for aggregating somatic mutation data across many samples.

Key functions
~~~~~~~~~~~~~
* :func:`mutation_burden`      – compute tumour mutation burden (TMB) per sample
* :func:`gene_alteration_matrix` – binary sample × gene alteration matrix
* :func:`co_occurrence`        – pairwise co-occurrence / mutual-exclusivity test
* :func:`cohort_summary`       – high-level statistics for an entire cohort
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


# ---------------------------------------------------------------------------
# Tumour mutation burden
# ---------------------------------------------------------------------------


def mutation_burden(
    df: pd.DataFrame,
    sample_col: str = "Tumor_Sample_Barcode",
    callable_mb: float = 30.0,
) -> pd.DataFrame:
    """Compute tumour mutation burden (mutations per megabase) per sample.

    Parameters
    ----------
    df:
        MAF-like DataFrame.  Each row represents one somatic mutation.
    sample_col:
        Column that identifies samples.
    callable_mb:
        Size of the callable / exome target in megabases used to normalise
        mutation counts (default 30 Mb, a typical WES callable region).

    Returns
    -------
    pandas.DataFrame
        Columns: ``[sample_col, "n_mutations", "TMB"]`` where TMB is
        mutations per Mb.
    """
    counts = df[sample_col].value_counts().rename("n_mutations").reset_index()
    counts.columns = [sample_col, "n_mutations"]
    counts["TMB"] = counts["n_mutations"] / callable_mb
    return counts.sort_values("TMB", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Alteration matrix
# ---------------------------------------------------------------------------


def gene_alteration_matrix(
    df: pd.DataFrame,
    gene_col: str = "Hugo_Symbol",
    sample_col: str = "Tumor_Sample_Barcode",
    genes: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Build a binary sample × gene alteration matrix.

    Each cell is ``1`` if the sample has ≥ 1 mutation in that gene, else ``0``.

    Parameters
    ----------
    df:
        MAF-like DataFrame.
    gene_col:
        Column containing gene symbols.
    sample_col:
        Column containing sample identifiers.
    genes:
        Optional list of genes to include.  If ``None``, all genes present in
        *df* are used.

    Returns
    -------
    pandas.DataFrame
        Index = samples, columns = genes.
    """
    if genes is not None:
        df = df[df[gene_col].isin(genes)]

    matrix = (
        df.groupby([sample_col, gene_col])
        .size()
        .unstack(fill_value=0)
        .clip(upper=1)
    )
    if genes is not None:
        for g in genes:
            if g not in matrix.columns:
                matrix[g] = 0
        matrix = matrix[genes]
    return matrix


# ---------------------------------------------------------------------------
# Co-occurrence / mutual exclusivity
# ---------------------------------------------------------------------------


def co_occurrence(
    alteration_matrix: pd.DataFrame,
    gene_a: str,
    gene_b: str,
) -> dict:
    """Test pairwise co-occurrence or mutual exclusivity between two genes.

    Uses a two-sided Fisher's exact test on the 2×2 contingency table of
    samples that are mutated / wild-type in each gene.

    Parameters
    ----------
    alteration_matrix:
        Binary sample × gene matrix (as returned by
        :func:`gene_alteration_matrix`).
    gene_a:
        First gene symbol.
    gene_b:
        Second gene symbol.

    Returns
    -------
    dict with keys:
        * ``n_both``       – samples mutated in both genes
        * ``n_a_only``     – samples mutated only in *gene_a*
        * ``n_b_only``     – samples mutated only in *gene_b*
        * ``n_neither``    – samples wild-type in both genes
        * ``odds_ratio``   – Fisher's exact odds ratio
        * ``pvalue``       – two-sided p-value
        * ``tendency``     – ``'co-occurrence'``, ``'mutual_exclusivity'``, or ``'neutral'``
    """
    for g in (gene_a, gene_b):
        if g not in alteration_matrix.columns:
            raise KeyError(f"Gene '{g}' not found in alteration matrix.")

    a = alteration_matrix[gene_a].astype(bool)
    b = alteration_matrix[gene_b].astype(bool)

    n_both = (a & b).sum()
    n_a_only = (a & ~b).sum()
    n_b_only = (~a & b).sum()
    n_neither = (~a & ~b).sum()

    table = [[n_both, n_a_only], [n_b_only, n_neither]]
    odds_ratio, pvalue = fisher_exact(table, alternative="two-sided")

    if pvalue < 0.05:
        tendency = "co-occurrence" if odds_ratio > 1 else "mutual_exclusivity"
    else:
        tendency = "neutral"

    return {
        "n_both": int(n_both),
        "n_a_only": int(n_a_only),
        "n_b_only": int(n_b_only),
        "n_neither": int(n_neither),
        "odds_ratio": float(odds_ratio),
        "pvalue": float(pvalue),
        "tendency": tendency,
    }


# ---------------------------------------------------------------------------
# Cohort summary
# ---------------------------------------------------------------------------


def cohort_summary(
    df: pd.DataFrame,
    sample_col: str = "Tumor_Sample_Barcode",
    gene_col: str = "Hugo_Symbol",
    classification_col: str = "Variant_Classification",
) -> dict:
    """Return high-level statistics for an entire cohort.

    Parameters
    ----------
    df:
        MAF-like DataFrame.
    sample_col:
        Sample identifier column.
    gene_col:
        Gene symbol column.
    classification_col:
        Variant classification column.

    Returns
    -------
    dict with keys:
        * ``n_samples``           – number of unique samples
        * ``n_mutations``         – total mutation count
        * ``n_genes``             – number of unique genes mutated
        * ``median_tmb``          – median mutations per sample
        * ``top_classification``  – most common Variant_Classification
        * ``top_genes``           – top-5 most frequently mutated genes
    """
    n_samples = df[sample_col].nunique()
    n_mutations = len(df)
    n_genes = df[gene_col].nunique()
    per_sample = df[sample_col].value_counts()
    median_tmb = float(per_sample.median()) if len(per_sample) > 0 else 0.0
    top_class = df[classification_col].value_counts().idxmax() if n_mutations > 0 else None
    top_genes = df[gene_col].value_counts().head(5).index.tolist()

    return {
        "n_samples": n_samples,
        "n_mutations": n_mutations,
        "n_genes": n_genes,
        "median_tmb": median_tmb,
        "top_classification": top_class,
        "top_genes": top_genes,
    }


# ---------------------------------------------------------------------------
# Recurrence table
# ---------------------------------------------------------------------------


def recurrence_table(
    df: pd.DataFrame,
    gene_col: str = "Hugo_Symbol",
    sample_col: str = "Tumor_Sample_Barcode",
) -> pd.DataFrame:
    """Build a gene-level recurrence table.

    Parameters
    ----------
    df:
        MAF-like DataFrame.
    gene_col:
        Gene symbol column.
    sample_col:
        Sample identifier column.

    Returns
    -------
    pandas.DataFrame
        Columns: ``[gene_col, "n_mutations", "n_samples", "sample_frequency"]``
        sorted by ``n_samples`` descending.
    """
    n_total_samples = df[sample_col].nunique()

    grouped = df.groupby(gene_col).agg(
        n_mutations=(sample_col, "count"),
        n_samples=(sample_col, "nunique"),
    )
    grouped["sample_frequency"] = grouped["n_samples"] / n_total_samples
    return grouped.sort_values("n_samples", ascending=False).reset_index()

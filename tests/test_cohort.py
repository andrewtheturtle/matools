"""Tests for matools.cohort – cohort-level aggregation helpers."""

import pandas as pd
import pytest

from matools.cohort import (
    co_occurrence,
    cohort_summary,
    gene_alteration_matrix,
    mutation_burden,
    recurrence_table,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def maf_df():
    data = {
        "Hugo_Symbol": [
            "TP53", "KRAS", "TP53", "BRCA1", "PIK3CA", "EGFR",
            "TP53", "KRAS", "MYC",
        ],
        "Tumor_Sample_Barcode": [
            "S1", "S1", "S2", "S2", "S3", "S3",
            "S4", "S4", "S4",
        ],
        "Variant_Classification": [
            "Missense_Mutation", "Silent",
            "Nonsense_Mutation", "Frame_Shift_Del",
            "Missense_Mutation", "Splice_Site",
            "Missense_Mutation", "Missense_Mutation", "Missense_Mutation",
        ],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# mutation_burden
# ---------------------------------------------------------------------------

class TestMutationBurden:
    def test_returns_dataframe(self, maf_df):
        result = mutation_burden(maf_df)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, maf_df):
        result = mutation_burden(maf_df)
        assert "Tumor_Sample_Barcode" in result.columns
        assert "n_mutations" in result.columns
        assert "TMB" in result.columns

    def test_one_row_per_sample(self, maf_df):
        result = mutation_burden(maf_df)
        assert len(result) == maf_df["Tumor_Sample_Barcode"].nunique()

    def test_tmb_calculation(self, maf_df):
        result = mutation_burden(maf_df, callable_mb=10.0)
        s4 = result[result["Tumor_Sample_Barcode"] == "S4"].iloc[0]
        assert s4["n_mutations"] == 3
        assert abs(s4["TMB"] - 0.3) < 1e-9

    def test_sorted_by_tmb(self, maf_df):
        result = mutation_burden(maf_df)
        tmb_vals = result["TMB"].tolist()
        assert tmb_vals == sorted(tmb_vals, reverse=True)


# ---------------------------------------------------------------------------
# gene_alteration_matrix
# ---------------------------------------------------------------------------

class TestGeneAlterationMatrix:
    def test_returns_dataframe(self, maf_df):
        mat = gene_alteration_matrix(maf_df)
        assert isinstance(mat, pd.DataFrame)

    def test_binary_values(self, maf_df):
        mat = gene_alteration_matrix(maf_df)
        assert set(mat.values.flatten()).issubset({0, 1})

    def test_index_is_samples(self, maf_df):
        mat = gene_alteration_matrix(maf_df)
        assert set(mat.index) == set(maf_df["Tumor_Sample_Barcode"])

    def test_gene_filter(self, maf_df):
        mat = gene_alteration_matrix(maf_df, genes=["TP53", "KRAS"])
        assert list(mat.columns) == ["TP53", "KRAS"]

    def test_missing_gene_filled_with_zero(self, maf_df):
        mat = gene_alteration_matrix(maf_df, genes=["TP53", "NOTAREALG"])
        assert "NOTAREALG" in mat.columns
        assert mat["NOTAREALG"].sum() == 0

    def test_tp53_present_in_three_samples(self, maf_df):
        mat = gene_alteration_matrix(maf_df)
        assert mat["TP53"].sum() == 3


# ---------------------------------------------------------------------------
# co_occurrence
# ---------------------------------------------------------------------------

class TestCoOccurrence:
    def test_returns_dict(self, maf_df):
        mat = gene_alteration_matrix(maf_df)
        result = co_occurrence(mat, "TP53", "KRAS")
        assert isinstance(result, dict)

    def test_required_keys(self, maf_df):
        mat = gene_alteration_matrix(maf_df)
        result = co_occurrence(mat, "TP53", "KRAS")
        for key in ("n_both", "n_a_only", "n_b_only", "n_neither", "odds_ratio", "pvalue", "tendency"):
            assert key in result

    def test_pvalue_in_range(self, maf_df):
        mat = gene_alteration_matrix(maf_df)
        result = co_occurrence(mat, "TP53", "KRAS")
        assert 0.0 <= result["pvalue"] <= 1.0

    def test_missing_gene_raises(self, maf_df):
        mat = gene_alteration_matrix(maf_df)
        with pytest.raises(KeyError):
            co_occurrence(mat, "TP53", "DOESNOTEXIST")

    def test_contingency_counts_sum(self, maf_df):
        mat = gene_alteration_matrix(maf_df)
        result = co_occurrence(mat, "TP53", "KRAS")
        total = result["n_both"] + result["n_a_only"] + result["n_b_only"] + result["n_neither"]
        assert total == len(mat)


# ---------------------------------------------------------------------------
# cohort_summary
# ---------------------------------------------------------------------------

class TestCohortSummary:
    def test_returns_dict(self, maf_df):
        result = cohort_summary(maf_df)
        assert isinstance(result, dict)

    def test_n_samples(self, maf_df):
        result = cohort_summary(maf_df)
        assert result["n_samples"] == 4

    def test_n_mutations(self, maf_df):
        result = cohort_summary(maf_df)
        assert result["n_mutations"] == len(maf_df)

    def test_top_genes_list(self, maf_df):
        result = cohort_summary(maf_df)
        assert isinstance(result["top_genes"], list)
        assert len(result["top_genes"]) <= 5

    def test_median_tmb_positive(self, maf_df):
        result = cohort_summary(maf_df)
        assert result["median_tmb"] > 0


# ---------------------------------------------------------------------------
# recurrence_table
# ---------------------------------------------------------------------------

class TestRecurrenceTable:
    def test_returns_dataframe(self, maf_df):
        result = recurrence_table(maf_df)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self, maf_df):
        result = recurrence_table(maf_df)
        for col in ("Hugo_Symbol", "n_mutations", "n_samples", "sample_frequency"):
            assert col in result.columns

    def test_sorted_by_n_samples(self, maf_df):
        result = recurrence_table(maf_df)
        vals = result["n_samples"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_sample_frequency_in_range(self, maf_df):
        result = recurrence_table(maf_df)
        assert result["sample_frequency"].between(0, 1).all()

    def test_tp53_top_gene(self, maf_df):
        result = recurrence_table(maf_df)
        assert result.iloc[0]["Hugo_Symbol"] == "TP53"

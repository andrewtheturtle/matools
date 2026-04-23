"""Tests for matools.variants – filtering, annotation, and summary helpers."""

import pandas as pd
import pytest

from matools.variants import (
    NONSYNONYMOUS_CLASSIFICATIONS,
    add_mutation_class,
    add_vaf,
    filter_by_classification,
    filter_by_sample,
    filter_by_type,
    filter_by_vaf,
    filter_nonsynonymous,
    most_frequent_genes,
    mutation_spectrum,
    summarise_variants,
)


# ---------------------------------------------------------------------------
# Minimal MAF-like fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def maf_df():
    data = {
        "Hugo_Symbol": ["TP53", "KRAS", "TP53", "BRCA1", "PIK3CA", "EGFR"],
        "Tumor_Sample_Barcode": ["S1", "S1", "S2", "S2", "S3", "S3"],
        "Variant_Classification": [
            "Missense_Mutation",
            "Silent",
            "Nonsense_Mutation",
            "Frame_Shift_Del",
            "Missense_Mutation",
            "Splice_Site",
        ],
        "Variant_Type": ["SNP", "SNP", "SNP", "DEL", "SNP", "SNP"],
        "Reference_Allele": ["C", "G", "A", "G", "T", "C"],
        "Tumor_Seq_Allele2": ["T", "A", "T", "G", "A", "T"],
        "t_ref_count": [80, 90, 50, 60, 70, 30],
        "t_alt_count": [20, 10, 50, 40, 30, 70],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Filtering tests
# ---------------------------------------------------------------------------

class TestFilterByClassification:
    def test_keep(self, maf_df):
        result = filter_by_classification(maf_df, keep=["Missense_Mutation"])
        assert all(result["Variant_Classification"] == "Missense_Mutation")
        assert len(result) == 2

    def test_drop(self, maf_df):
        result = filter_by_classification(maf_df, drop=["Silent"])
        assert "Silent" not in result["Variant_Classification"].values
        assert len(result) == 5

    def test_keep_and_drop(self, maf_df):
        result = filter_by_classification(
            maf_df,
            keep=["Missense_Mutation", "Silent"],
            drop=["Silent"],
        )
        assert len(result) == 2
        assert "Silent" not in result["Variant_Classification"].values

    def test_empty_result(self, maf_df):
        result = filter_by_classification(maf_df, keep=["Nonsense_Mutation_XYZ"])
        assert len(result) == 0

    def test_index_reset(self, maf_df):
        result = filter_by_classification(maf_df, keep=["Missense_Mutation"])
        assert list(result.index) == list(range(len(result)))


class TestFilterByType:
    def test_keep_snp(self, maf_df):
        result = filter_by_type(maf_df, keep=["SNP"])
        assert all(result["Variant_Type"] == "SNP")

    def test_drop_del(self, maf_df):
        result = filter_by_type(maf_df, drop=["DEL"])
        assert "DEL" not in result["Variant_Type"].values


class TestFilterNonsynonymous:
    def test_removes_silent(self, maf_df):
        result = filter_nonsynonymous(maf_df)
        assert "Silent" not in result["Variant_Classification"].values

    def test_keeps_missense(self, maf_df):
        result = filter_nonsynonymous(maf_df)
        assert "Missense_Mutation" in result["Variant_Classification"].values

    def test_all_retained_are_nonsynonymous(self, maf_df):
        result = filter_nonsynonymous(maf_df)
        assert all(v in NONSYNONYMOUS_CLASSIFICATIONS for v in result["Variant_Classification"])


class TestFilterByVaf:
    def test_requires_vaf_column(self, maf_df):
        with pytest.raises(KeyError):
            filter_by_vaf(maf_df)

    def test_min_vaf(self, maf_df):
        df = add_vaf(maf_df)
        result = filter_by_vaf(df, min_vaf=0.5)
        assert all(result["VAF"] >= 0.5)

    def test_max_vaf(self, maf_df):
        df = add_vaf(maf_df)
        result = filter_by_vaf(df, max_vaf=0.3)
        assert all(result["VAF"] <= 0.3)


class TestFilterBySample:
    def test_keeps_only_specified_samples(self, maf_df):
        result = filter_by_sample(maf_df, ["S1"])
        assert set(result["Tumor_Sample_Barcode"]) == {"S1"}

    def test_multiple_samples(self, maf_df):
        result = filter_by_sample(maf_df, ["S1", "S3"])
        assert set(result["Tumor_Sample_Barcode"]) == {"S1", "S3"}


# ---------------------------------------------------------------------------
# Annotation tests
# ---------------------------------------------------------------------------

class TestAddVaf:
    def test_creates_vaf_column(self, maf_df):
        df = add_vaf(maf_df)
        assert "VAF" in df.columns

    def test_vaf_values_in_range(self, maf_df):
        df = add_vaf(maf_df)
        assert df["VAF"].between(0, 1).all()

    def test_custom_column_name(self, maf_df):
        df = add_vaf(maf_df, vaf_col="tumour_vaf")
        assert "tumour_vaf" in df.columns

    def test_does_not_modify_original(self, maf_df):
        original_cols = list(maf_df.columns)
        add_vaf(maf_df)
        assert list(maf_df.columns) == original_cols

    def test_vaf_calculation(self):
        df = pd.DataFrame({"t_ref_count": [80], "t_alt_count": [20]})
        result = add_vaf(df)
        assert abs(result.loc[0, "VAF"] - 0.2) < 1e-9


class TestAddMutationClass:
    def test_adds_column(self, maf_df):
        df = add_mutation_class(maf_df)
        assert "Mutation_Class" in df.columns

    def test_missense_is_nonsynonymous(self, maf_df):
        df = add_mutation_class(maf_df)
        missense = df[df["Variant_Classification"] == "Missense_Mutation"]
        assert all(missense["Mutation_Class"] == "nonsynonymous")

    def test_silent_is_synonymous(self, maf_df):
        df = add_mutation_class(maf_df)
        silent = df[df["Variant_Classification"] == "Silent"]
        assert all(silent["Mutation_Class"] == "synonymous")


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------

class TestSummariseVariants:
    def test_returns_dataframe(self, maf_df):
        result = summarise_variants(maf_df)
        assert isinstance(result, pd.DataFrame)

    def test_groups_by_sample(self, maf_df):
        result = summarise_variants(maf_df, by="Tumor_Sample_Barcode")
        assert set(result["Tumor_Sample_Barcode"]) == {"S1", "S2", "S3"}

    def test_total_column(self, maf_df):
        result = summarise_variants(maf_df)
        assert "Total" in result.columns


class TestMostFrequentGenes:
    def test_returns_dataframe(self, maf_df):
        result = most_frequent_genes(maf_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_count_column(self, maf_df):
        result = most_frequent_genes(maf_df)
        assert "count" in result.columns

    def test_sorted_descending(self, maf_df):
        result = most_frequent_genes(maf_df)
        counts = result["count"].tolist()
        assert counts == sorted(counts, reverse=True)

    def test_top_n(self, maf_df):
        result = most_frequent_genes(maf_df, n=2)
        assert len(result) <= 2

    def test_by_mutations(self, maf_df):
        result = most_frequent_genes(maf_df, count_method="mutations")
        assert result.iloc[0]["Hugo_Symbol"] == "TP53"

    def test_invalid_count_method(self, maf_df):
        with pytest.raises(ValueError):
            most_frequent_genes(maf_df, count_method="invalid")


class TestMutationSpectrum:
    def test_returns_series(self, maf_df):
        snp_df = filter_by_type(maf_df, keep=["SNP"])
        result = mutation_spectrum(snp_df)
        assert isinstance(result, pd.Series)

    def test_canonical_keys(self, maf_df):
        snp_df = filter_by_type(maf_df, keep=["SNP"])
        result = mutation_spectrum(snp_df)
        assert set(result.index) == {"C>A", "C>G", "C>T", "T>A", "T>C", "T>G"}

    def test_counts_non_negative(self, maf_df):
        snp_df = filter_by_type(maf_df, keep=["SNP"])
        result = mutation_spectrum(snp_df)
        assert (result >= 0).all()

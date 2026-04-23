"""Tests for matools.utils – general-purpose helpers."""

from pathlib import Path

import pandas as pd
import pytest

from matools.utils import (
    chunks,
    complement_strand,
    filter_genes,
    flatten,
    gc_content,
    load_gene_list,
    overlap_regions,
    parse_genomic_region,
)


# ---------------------------------------------------------------------------
# parse_genomic_region
# ---------------------------------------------------------------------------

class TestParseGenomicRegion:
    def test_basic(self):
        result = parse_genomic_region("chr7:140453136-140453137")
        assert result["chrom"] == "chr7"
        assert result["start"] == 140453136
        assert result["end"] == 140453137

    def test_without_chr_prefix(self):
        result = parse_genomic_region("17:7674220-7674221")
        assert result["chrom"] == "17"

    def test_commas_in_coords(self):
        result = parse_genomic_region("chr1:1,000-2,000")
        assert result["start"] == 1000
        assert result["end"] == 2000

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_genomic_region("notaregion")

    def test_returns_dict_with_correct_keys(self):
        result = parse_genomic_region("chr1:100-200")
        assert set(result.keys()) == {"chrom", "start", "end"}


# ---------------------------------------------------------------------------
# overlap_regions
# ---------------------------------------------------------------------------

@pytest.fixture()
def query_bed():
    return pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr2"],
        "chromStart": [100, 500, 200],
        "chromEnd": [300, 700, 400],
    })


@pytest.fixture()
def subject_bed():
    return pd.DataFrame({
        "chrom": ["chr1", "chr3"],
        "chromStart": [200, 0],
        "chromEnd": [600, 100],
    })


class TestOverlapRegions:
    def test_returns_dataframe(self, query_bed, subject_bed):
        result = overlap_regions(query_bed, subject_bed)
        assert isinstance(result, pd.DataFrame)

    def test_chr2_not_returned(self, query_bed, subject_bed):
        result = overlap_regions(query_bed, subject_bed)
        assert "chr2" not in result["chrom"].values

    def test_overlapping_rows_found(self, query_bed, subject_bed):
        result = overlap_regions(query_bed, subject_bed)
        assert len(result) >= 1

    def test_no_overlap_returns_empty(self, query_bed):
        subject = pd.DataFrame({
            "chrom": ["chr9"],
            "chromStart": [0],
            "chromEnd": [100],
        })
        result = overlap_regions(query_bed, subject)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# complement_strand
# ---------------------------------------------------------------------------

class TestComplementStrand:
    def test_basic(self):
        assert complement_strand("ATCG") == "CGAT"

    def test_reverse_complement(self):
        assert complement_strand("AACCGGTT") == "AACCGGTT"

    def test_n_preserved(self):
        result = complement_strand("ANG")
        assert "N" in result

    def test_case_preserved(self):
        result = complement_strand("atcg")
        assert result == result.lower()


# ---------------------------------------------------------------------------
# gc_content
# ---------------------------------------------------------------------------

class TestGcContent:
    def test_all_gc(self):
        assert gc_content("GCGCGC") == 1.0

    def test_all_at(self):
        assert gc_content("ATATAT") == 0.0

    def test_mixed(self):
        assert abs(gc_content("ACGT") - 0.5) < 1e-9

    def test_empty_sequence(self):
        import math
        assert math.isnan(gc_content("NNNNN"))

    def test_case_insensitive(self):
        assert gc_content("gcgc") == 1.0


# ---------------------------------------------------------------------------
# load_gene_list
# ---------------------------------------------------------------------------

class TestLoadGeneList:
    def test_reads_genes(self, tmp_path):
        p = tmp_path / "genes.txt"
        p.write_text("# cancer genes\nTP53\nKRAS\nBRCA1\n")
        result = load_gene_list(p)
        assert result == ["TP53", "KRAS", "BRCA1"]

    def test_skips_comments(self, tmp_path):
        p = tmp_path / "genes.txt"
        p.write_text("# comment\nTP53\n")
        result = load_gene_list(p)
        assert result == ["TP53"]

    def test_strips_whitespace(self, tmp_path):
        p = tmp_path / "genes.txt"
        p.write_text("  TP53  \n")
        result = load_gene_list(p)
        assert result == ["TP53"]


# ---------------------------------------------------------------------------
# filter_genes
# ---------------------------------------------------------------------------

class TestFilterGenes:
    def test_keeps_specified_genes(self):
        df = pd.DataFrame({"Hugo_Symbol": ["TP53", "KRAS", "BRCA1"]})
        result = filter_genes(df, ["TP53", "KRAS"])
        assert set(result["Hugo_Symbol"]) == {"TP53", "KRAS"}

    def test_empty_gene_list(self):
        df = pd.DataFrame({"Hugo_Symbol": ["TP53", "KRAS"]})
        result = filter_genes(df, [])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# chunks
# ---------------------------------------------------------------------------

class TestChunks:
    def test_even_split(self):
        assert list(chunks(range(6), 2)) == [[0, 1], [2, 3], [4, 5]]

    def test_uneven_split(self):
        result = list(chunks(range(10), 3))
        assert result[-1] == [9]
        assert len(result) == 4

    def test_size_larger_than_iterable(self):
        assert list(chunks([1, 2], 10)) == [[1, 2]]

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            list(chunks([1, 2, 3], 0))

    def test_empty_iterable(self):
        assert list(chunks([], 3)) == []


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------

class TestFlatten:
    def test_basic(self):
        assert flatten([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]

    def test_empty_sublists(self):
        assert flatten([[], [], []]) == []

    def test_single_list(self):
        assert flatten([[1, 2, 3]]) == [1, 2, 3]

# matools

A centralized Python toolkit for cancer genomics analyses — helper functions and
reusable code snippets for working with somatic mutation data across different
servers and databases.

---

## Installation

```bash
pip install -e ".[dev]"   # editable install with dev/test extras
```

**Requirements:** Python ≥ 3.9, pandas ≥ 1.5, numpy ≥ 1.23, scipy ≥ 1.10

---

## Package layout

```
matools/
├── io.py        – read/write MAF, VCF, BED, and generic TSV files
├── variants.py  – variant filtering, VAF annotation, mutation summaries
├── cohort.py    – cohort-level TMB, alteration matrices, co-occurrence tests
└── utils.py     – genomic-region parsing, sequence utilities, gene-list helpers
```

---

## Quick-start examples

### Reading a MAF file

```python
from matools.io import read_maf

df = read_maf("mutations.maf")          # plain or .gz
df = read_maf("mutations.maf.gz")       # gzip transparent
print(df.head())
```

### Filtering to non-synonymous mutations

```python
from matools.variants import filter_nonsynonymous, add_vaf

ns = filter_nonsynonymous(df)           # keeps Missense, Nonsense, Splice_Site, …
ns = add_vaf(ns)                        # appends a VAF column from t_ref/t_alt counts
```

### Most frequently mutated genes

```python
from matools.variants import most_frequent_genes

top = most_frequent_genes(ns, n=20, count_method="samples")
print(top)
```

### Tumour mutation burden

```python
from matools.cohort import mutation_burden

tmb = mutation_burden(ns, callable_mb=38.0)
print(tmb.head())
```

### Sample × gene alteration matrix

```python
from matools.cohort import gene_alteration_matrix, co_occurrence

mat = gene_alteration_matrix(ns, genes=["TP53", "KRAS", "BRCA1"])
result = co_occurrence(mat, "TP53", "KRAS")
print(result["tendency"], result["pvalue"])
```

### Reading VCF / BED files

```python
from matools.io import read_vcf, read_bed

headers, vcf_df = read_vcf("calls.vcf")
bed_df = read_bed("targets.bed", extra_cols=["name", "score"])
```

### Genomic-region and sequence utilities

```python
from matools.utils import parse_genomic_region, complement_strand, gc_content

region = parse_genomic_region("chr7:140,453,136-140,453,137")
# {'chrom': 'chr7', 'start': 140453136, 'end': 140453137}

rc = complement_strand("ATCGATCG")   # → "CGATCGAT"
gc = gc_content("ATCG")              # → 0.5
```

---

## Module reference

### `matools.io`

| Function | Description |
|---|---|
| `read_maf(path)` | Read MAF (plain or .gz) → DataFrame |
| `write_maf(df, path)` | Write DataFrame as MAF |
| `read_vcf(path)` | Read VCF → (header_lines, DataFrame) |
| `read_bed(path)` | Read BED → DataFrame |
| `read_tsv(path)` | Read generic TSV, skipping `#` comments |
| `write_tsv(df, path)` | Write DataFrame as TSV |

### `matools.variants`

| Function | Description |
|---|---|
| `filter_nonsynonymous(df)` | Keep only protein-altering mutations |
| `filter_by_classification(df, keep, drop)` | Filter by Variant_Classification |
| `filter_by_type(df, keep, drop)` | Filter by Variant_Type (SNP/INS/DEL/…) |
| `filter_by_vaf(df, min_vaf, max_vaf)` | Filter by variant-allele fraction |
| `filter_by_sample(df, samples)` | Keep rows matching given sample IDs |
| `add_vaf(df)` | Compute VAF from t_ref/t_alt counts |
| `add_mutation_class(df)` | Add simplified nonsynonymous/synonymous label |
| `summarise_variants(df, by)` | Pivot table of mutation counts |
| `most_frequent_genes(df, n)` | Top-N genes by sample or mutation count |
| `mutation_spectrum(df)` | SNP substitution spectrum (C>A, C>G, …) |

### `matools.cohort`

| Function | Description |
|---|---|
| `mutation_burden(df, callable_mb)` | TMB (mut/Mb) per sample |
| `gene_alteration_matrix(df, genes)` | Binary sample × gene alteration matrix |
| `co_occurrence(matrix, geneA, geneB)` | Fisher's exact co-occurrence/exclusivity test |
| `cohort_summary(df)` | High-level cohort statistics dict |
| `recurrence_table(df)` | Gene-level recurrence (n_samples, sample_frequency) |

### `matools.utils`

| Function | Description |
|---|---|
| `parse_genomic_region(region)` | Parse `"chr1:1000-2000"` → dict |
| `overlap_regions(query, subject)` | BED-style interval overlap filter |
| `complement_strand(seq)` | Reverse complement a DNA sequence |
| `gc_content(seq)` | GC fraction of a DNA sequence |
| `load_gene_list(path)` | Read a plain-text gene list |
| `filter_genes(df, genes)` | Filter DataFrame to rows matching gene list |
| `chunks(iterable, size)` | Split iterable into fixed-size batches |
| `flatten(nested)` | Flatten one level of nesting |

---

## Running tests

```bash
pytest tests/ -v
```

---

## Contributing

Add new helper functions in the appropriate module (`io`, `variants`, `cohort`,
or `utils`).  Add a corresponding test in `tests/` and verify with `pytest`
before opening a pull request.


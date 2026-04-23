"""
matools – cancer genomics helper functions.

Modules
-------
io        : read/write common genomics file formats (MAF, VCF, BED, TSV)
variants  : variant filtering, annotation, and summarization
cohort    : cohort-level mutation-frequency and gene-level aggregation
utils     : general-purpose helpers (gene lists, genomic-coordinate utilities)
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("matools")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["io", "variants", "cohort", "utils"]

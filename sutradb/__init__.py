"""
SutraDB (सूत्र DB) - High-performance, zero-dependency hybrid vector search
and BM25 lexical engine in pure Python.
"""

from sutradb.core import SutraDB, Collection, Document, SearchResult
from sutradb.distance import Metric
from sutradb.filters import FilterEngine
from sutradb.bm25 import BM25Index
from sutradb.fusion import reciprocal_rank_fusion, linear_score_fusion
from sutradb.ann import IVFIndex, IVFConfig, auto_n_lists

__version__ = "1.0.0"
__all__ = [
    "SutraDB",
    "Collection",
    "Document",
    "SearchResult",
    "Metric",
    "FilterEngine",
    "BM25Index",
    "reciprocal_rank_fusion",
    "linear_score_fusion",
    "IVFIndex",
    "IVFConfig",
    "auto_n_lists",
]

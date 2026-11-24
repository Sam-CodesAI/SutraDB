"""
SutraDB (सूत्र DB) - High-performance, zero-dependency hybrid vector search
and BM25 lexical engine in pure Python & Rust.
"""

from sutradb.core import SutraDB, Collection, Document, SearchResult, NativeCollection
try:
    from sutradb.client import SutraClient
except ImportError:
    SutraClient = None
from sutradb.distance import Metric
from sutradb.filters import FilterEngine
from sutradb.bm25 import BM25Index
from sutradb.fusion import reciprocal_rank_fusion, linear_score_fusion
from sutradb.ann import IVFIndex, IVFConfig, auto_n_lists
from sutradb.integrations.langchain import SutraDBVectorStore
from sutradb.integrations.llamaindex import SutraDBLlamaIndexVectorStore
from sutradb.quantization import BQVector, SQ8Vector, PQVector, PQCodebook
from sutradb.nlp import Analyzer, AnalyzerConfig, PorterStemmer
from sutradb.embeddings import (
    EmbeddingFunction,
    DefaultEmbeddingFunction,
    OpenAIEmbeddingFunction,
    FastEmbedFunction,
    HuggingFaceEmbeddingFunction,
)
from sutradb.cluster import (
    ConsistentHashRing,
    DistributedCollection,
    DistributedCluster,
)

__version__ = "5.1.0"
__all__ = [
    "SutraDB",
    "Collection",
    "Document",
    "SearchResult",
    "NativeCollection",
    "SutraClient",
    "SutraDBVectorStore",
    "SutraDBLlamaIndexVectorStore",
    "BQVector",
    "SQ8Vector",
    "PQVector",
    "PQCodebook",
    "Analyzer",
    "AnalyzerConfig",
    "PorterStemmer",
    "Metric",
    "FilterEngine",
    "BM25Index",
    "reciprocal_rank_fusion",
    "linear_score_fusion",
    "IVFIndex",
    "IVFConfig",
    "auto_n_lists",
    "EmbeddingFunction",
    "DefaultEmbeddingFunction",
    "OpenAIEmbeddingFunction",
    "FastEmbedFunction",
    "HuggingFaceEmbeddingFunction",
    "ConsistentHashRing",
    "DistributedCollection",
    "DistributedCluster",
]

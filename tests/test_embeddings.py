"""
Tests for SutraDB automated embedding ingestion, model pipeline, and text search.
Verifies DefaultEmbeddingFunction, OpenAIEmbeddingFunction, Collection.add_texts,
Collection.search, and auto-vectorization in Collection.insert.
"""

import json
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from sutradb import (
    SutraDB,
    Collection,
    Document,
    Metric,
    EmbeddingFunction,
    DefaultEmbeddingFunction,
    OpenAIEmbeddingFunction,
    FastEmbedFunction,
    HuggingFaceEmbeddingFunction,
)


class TestDefaultEmbeddingFunction:
    """Verifies the deterministic zero-dependency default embedding generator."""

    def test_output_dimension_and_norm(self):
        ef = DefaultEmbeddingFunction(dimension=384)
        texts = ["hello world", "sutradb vector database", "artificial intelligence"]
        vecs = ef(texts)

        assert len(vecs) == 3
        for v in vecs:
            assert len(v) == 384
            norm = np.linalg.norm(np.array(v))
            assert pytest.approx(norm, abs=1e-5) == 1.0

    def test_custom_dimension(self):
        ef = DefaultEmbeddingFunction(dimension=128)
        assert ef.dimension == 128
        vecs = ef(["quick brown fox"])
        assert len(vecs[0]) == 128

    def test_deterministic_consistency(self):
        ef1 = DefaultEmbeddingFunction(seed=42)
        ef2 = DefaultEmbeddingFunction(seed=42)

        v1 = ef1(["reproducible deterministic embeddings"])[0]
        v2 = ef2(["reproducible deterministic embeddings"])[0]
        assert v1 == v2

        ef_diff_seed = DefaultEmbeddingFunction(seed=99)
        v3 = ef_diff_seed(["reproducible deterministic embeddings"])[0]
        assert v1 != v3

    def test_different_texts_produce_different_vectors(self):
        ef = DefaultEmbeddingFunction()
        v1 = np.array(ef.embed_query("quantum computing"))
        v2 = np.array(ef.embed_query("gardening and agriculture"))

        similarity = float(np.dot(v1, v2))
        assert similarity < 0.95

    def test_empty_and_whitespace_texts(self):
        ef = DefaultEmbeddingFunction(dimension=64)
        vecs = ef(["", "   ", "\n\t"])
        assert len(vecs) == 3
        for v in vecs:
            assert len(v) == 64
            norm = np.linalg.norm(np.array(v))
            assert pytest.approx(norm, abs=1e-5) == 1.0

    def test_langchain_interface_compatibility(self):
        ef = DefaultEmbeddingFunction(dimension=64)
        # embed_documents
        doc_vecs = ef.embed_documents(["doc one", "doc two"])
        assert len(doc_vecs) == 2
        assert len(doc_vecs[0]) == 64
        # embed_query
        q_vec = ef.embed_query("query text")
        assert len(q_vec) == 64

    def test_encode_sparse(self):
        ef = DefaultEmbeddingFunction()
        sp = ef.encode_sparse(["vector search in rust and python"])
        assert len(sp) == 1
        assert isinstance(sp[0], dict)
        assert len(sp[0]) > 0
        for k, v in sp[0].items():
            assert isinstance(k, int)
            assert v >= 1.0

    def test_encode_colbert(self):
        ef = DefaultEmbeddingFunction(dimension=64)
        cb = ef.encode_colbert(["colbert multi vector late interaction"])
        assert len(cb) == 1
        # Token count up to 32 tokens
        assert len(cb[0]) == 5
        for tok_vec in cb[0]:
            assert len(tok_vec) == 64
            norm = np.linalg.norm(np.array(tok_vec))
            assert pytest.approx(norm, abs=1e-5) == 1.0


class TestOpenAIEmbeddingFunction:
    """Verifies OpenAI embedding provider wrapper and error handling."""

    def test_missing_api_key_raises_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ef = OpenAIEmbeddingFunction(api_key=None)
        with pytest.raises(ValueError, match="OpenAI API key must be passed"):
            ef(["test query"])

    def test_dimension_defaults(self):
        ef_small = OpenAIEmbeddingFunction(api_key="sk-test", model_name="text-embedding-3-small")
        assert ef_small.dimension == 1536

        ef_large = OpenAIEmbeddingFunction(api_key="sk-test", model_name="text-embedding-3-large")
        assert ef_large.dimension == 3072

        ef_custom = OpenAIEmbeddingFunction(api_key="sk-test", dimension=512)
        assert ef_custom.dimension == 512

    def test_mocked_http_request(self):
        ef = OpenAIEmbeddingFunction(api_key="sk-dummy-key", dimension=3)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
            ]
        }

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            embeddings = ef(["first text", "second text"])

            assert len(embeddings) == 2
            # Notice sorted by index: 0 first, then 1
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]

            # Verify post call arguments
            mock_client.post.assert_called_once()
            call_kwargs = mock_client.post.call_args[1]
            assert call_kwargs["json"]["input"] == ["first text", "second text"]
            assert call_kwargs["json"]["dimensions"] == 3


class TestOptionalEmbeddingProviders:
    """Verifies clean ImportError handling for optional dependencies."""

    def test_fastembed_import_error(self):
        with patch.dict("sys.modules", {"fastembed": None}):
            with pytest.raises(ImportError, match="fastembed is required"):
                FastEmbedFunction()

    def test_huggingface_import_error(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers is required"):
                HuggingFaceEmbeddingFunction()


class TestCollectionAddTextsAndSearch:
    """Verifies end-to-end raw text ingestion and direct text search."""

    def test_add_texts_and_text_search(self):
        col = Collection("kb_articles", dimension=384, metric=Metric.COSINE)
        texts = [
            "Rust provides zero-cost abstractions, memory safety without garbage collection.",
            "Python is widely used in machine learning, data science, and AI pipelines.",
            "SutraDB is an industrial vector database written in Rust and Python.",
            "Photosynthesis is the process used by plants to convert sunlight into energy.",
        ]
        metas = [
            {"topic": "systems", "lang": "rust"},
            {"topic": "ml", "lang": "python"},
            {"topic": "database", "lang": "multi"},
            {"topic": "biology", "lang": "nature"},
        ]

        ids = col.add_texts(texts=texts, metadatas=metas)
        assert len(ids) == 4
        assert len(col.documents) == 4
        assert col.vectors.shape == (4, 384)

        # Search directly with query text
        results = col.search(query="vector database in rust", top_k=2)
        assert len(results) == 2
        # SutraDB or Rust document should rank top
        top_topics = [r.metadata["topic"] for r in results]
        assert "database" in top_topics or "systems" in top_topics

    def test_add_texts_single_and_custom_ids(self):
        col = Collection("single_test", dimension=128)
        ids = col.add_texts(texts="Single text entry", ids="doc-custom-1", metadatas={"author": "alice"})
        assert ids == ["doc-custom-1"]
        assert col.documents[0].id == "doc-custom-1"
        assert col.documents[0].metadata["author"] == "alice"

    def test_add_texts_validation_mismatches(self):
        col = Collection("validation_test", dimension=64)
        with pytest.raises(ValueError, match="Length of metadatas"):
            col.add_texts(texts=["a", "b"], metadatas=[{"k": "v"}])

        with pytest.raises(ValueError, match="Length of ids"):
            col.add_texts(texts=["a", "b"], ids=["id1"])

    def test_add_texts_with_namespaces(self):
        col = Collection("multi_tenant", dimension=128)
        col.add_texts(
            texts=["User 1 private invoice record", "User 1 receipt"],
            namespace="tenant_100",
        )
        col.add_texts(
            texts=["User 2 confidential tax report"],
            namespace="tenant_200",
        )

        res_t1 = col.search("User 1 invoice", namespace="tenant_100", top_k=5)
        assert len(res_t1) == 2
        for r in res_t1:
            assert r.namespace == "tenant_100"

        res_t2 = col.search("tax report", namespace="tenant_200", top_k=5)
        assert len(res_t2) == 1
        assert res_t2[0].namespace == "tenant_200"

        # Verifies tenant isolation: searching tenant_100 never leaks tenant_200 records
        res_cross = col.search("confidential tax report", namespace="tenant_100", top_k=5)
        for r in res_cross:
            assert r.namespace == "tenant_100"
            assert "User 2" not in r.text

    def test_search_with_metadata_filter(self):
        col = Collection("filtered_text_search", dimension=128)
        col.add_texts(
            texts=[
                "High performance GPU server clustering",
                "Low latency edge gateway routing",
                "High performance database caching",
            ],
            metadatas=[
                {"category": "compute", "price": 500},
                {"category": "network", "price": 100},
                {"category": "compute", "price": 200},
            ]
        )

        # Filter category == compute AND price <= 300
        results = col.search(
            query="high performance",
            filter={"category": "compute", "price": {"$lte": 300}},
            top_k=5
        )
        assert len(results) == 1
        assert results[0].metadata["price"] == 200

    def test_insert_auto_vectorization_when_vector_missing(self):
        col = Collection("auto_vec_insert", dimension=64)

        # Insert Document instances with no vector provided
        doc1 = Document(id="d1", text="Machine learning model serving")
        doc2 = Document(id="d2", text="Realtime database streaming")
        col.insert([doc1, doc2])

        assert col.vectors.shape == (2, 64)
        assert doc1.vector is not None
        assert doc2.vector is not None
        assert doc1.sparse_vector is not None
        assert doc1.colbert_tokens is not None

        # Insert dict with no vector
        col.insert([{"id": "d3", "text": "Continuous delivery pipeline"}])
        assert col.vectors.shape == (3, 64)
        assert len(col.documents) == 3


class TestSutraDBManagerEmbeddingIntegration:
    """Verifies SutraDB manager collection creation with custom embedding functions."""

    def test_create_collection_with_embedding_function(self):
        db = SutraDB()
        custom_ef = DefaultEmbeddingFunction(dimension=256)

        col = db.create_collection("products", embedding_function=custom_ef)
        assert col.dimension == 256
        assert col.embedding_function is custom_ef

        col.add_texts(["Smartphone with OLED screen", "Wireless noise-canceling headphones"])
        assert col.vectors.shape == (2, 256)

        results = col.search("headphones audio sound", top_k=1)
        assert len(results) == 1
        assert "headphones" in results[0].text.lower()

    def test_get_or_create_collection_with_embedding_function(self):
        db = SutraDB()
        custom_ef = DefaultEmbeddingFunction(dimension=128)

        col1 = db.get_or_create_collection("analytics", embedding_function=custom_ef)
        assert col1.dimension == 128

        col2 = db.get_or_create_collection("analytics")
        assert col2 is col1

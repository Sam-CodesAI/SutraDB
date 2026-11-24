"""
Unit and integration tests for SutraDB LangChain VectorStore connector.
"""

import pytest
import shutil
from pathlib import Path
from typing import List

from sutradb.integrations.langchain import SutraDBVectorStore, Document


class MockEmbeddings:
    """Mock embeddings generating deterministic synthetic vectors."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        # Return deterministic 3-dimensional vector based on keywords
        t = text.lower()
        if "rust" in t or "performance" in t:
            return [1.0, 0.1, 0.0]
        elif "python" in t or "rag" in t:
            return [0.1, 1.0, 0.0]
        else:
            return [0.0, 0.1, 1.0]


@pytest.fixture
def temp_dir(tmp_path):
    d = tmp_path / "langchain_sutra_data"
    d.mkdir(parents=True, exist_ok=True)
    yield str(d)
    shutil.rmtree(str(d), ignore_errors=True)


def test_langchain_vectorstore_lifecycle(temp_dir):
    embeddings = MockEmbeddings()
    store = SutraDBVectorStore(
        collection_name="test_lc",
        dimension=3,
        embedding_function=embeddings,
        persist_directory=temp_dir,
        hybrid=True,
    )

    # 1. Add texts
    texts = [
        "Rust core vector database with zero-copy mmap.",
        "Python agent framework with RAG capabilities.",
        "General database storage with durability.",
    ]
    metadatas = [
        {"lang": "rust", "tier": "backend"},
        {"lang": "python", "tier": "agent"},
        {"lang": "sql", "tier": "storage"},
    ]
    ids = ["doc_rust", "doc_python", "doc_sql"]

    returned_ids = store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    assert returned_ids == ids
    assert store.collection.count() == 3

    # 2. Similarity search
    results = store.similarity_search("high performance rust engine", k=2)
    assert len(results) == 2
    assert results[0].id == "doc_rust"
    assert "Rust core" in results[0].page_content
    assert results[0].metadata["lang"] == "rust"

    # 3. Similarity search with score
    scored_results = store.similarity_search_with_score("python agent workflow", k=1)
    assert len(scored_results) == 1
    doc, score = scored_results[0]
    assert doc.id == "doc_python"
    assert score > 0.0

    # 4. Filtered search
    filtered = store.similarity_search("database", k=3, filter={"lang": "sql"})
    assert len(filtered) == 1
    assert filtered[0].id == "doc_sql"

    # 5. Maximal Marginal Relevance (MMR) search
    mmr_results = store.max_marginal_relevance_search("rust performance", k=2, fetch_k=3, lambda_mult=0.5)
    assert len(mmr_results) == 2
    assert mmr_results[0].id == "doc_rust"

    # 6. Delete
    deleted = store.delete(["doc_sql"])
    assert deleted is True
    assert store.collection.count() == 2

    # 7. as_retriever
    retriever = store.as_retriever(k=1)
    retrieved_docs = retriever.invoke("rust")
    assert len(retrieved_docs) >= 1
    assert retrieved_docs[0].id == "doc_rust"


def test_langchain_from_documents(temp_dir):
    embeddings = MockEmbeddings()
    docs = [
        Document(page_content="Vector indexing with HNSW", metadata={"topic": "hnsw"}, id="hnsw_1"),
        Document(page_content="Scalar quantization with SQ8", metadata={"topic": "sq8"}, id="sq8_1"),
    ]

    store = SutraDBVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="test_from_docs",
        persist_directory=temp_dir,
    )
    assert store.collection.count() == 2

    res = store.similarity_search("HNSW graph", k=1)
    assert len(res) == 1
    assert res[0].metadata["topic"] == "hnsw"

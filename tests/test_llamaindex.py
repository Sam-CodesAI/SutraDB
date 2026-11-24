"""
Unit and integration tests for SutraDB LlamaIndex VectorStore connector.
"""

import pytest
import shutil
from typing import List

from sutradb.integrations.llamaindex import (
    SutraDBLlamaIndexVectorStore,
    TextNode,
    VectorStoreQuery,
    VectorStoreQueryMode,
)


@pytest.fixture
def temp_dir(tmp_path):
    d = tmp_path / "llamaindex_sutra_data"
    d.mkdir(parents=True, exist_ok=True)
    yield str(d)
    shutil.rmtree(str(d), ignore_errors=True)


def test_llamaindex_vectorstore_lifecycle(temp_dir):
    store = SutraDBLlamaIndexVectorStore(
        collection_name="test_llama",
        dimension=3,
        persist_directory=temp_dir,
    )

    nodes = [
        TextNode(
            text="LlamaIndex query pipeline with SutraDB",
            id_="node_1",
            embedding=[1.0, 0.0, 0.0],
            metadata={"source": "docs", "author": "samarth"},
            ref_doc_id="parent_doc_1",
        ),
        TextNode(
            text="Distributed vector search and hybrid indexing",
            id_="node_2",
            embedding=[0.0, 1.0, 0.0],
            metadata={"source": "blog", "author": "samarth"},
            ref_doc_id="parent_doc_1",
        ),
        TextNode(
            text="Lexical keyword matching with BM25 Okapi",
            id_="node_3",
            embedding=[0.0, 0.0, 1.0],
            metadata={"source": "paper", "author": "researcher"},
            ref_doc_id="parent_doc_2",
        ),
    ]

    # 1. Add nodes
    added_ids = store.add(nodes)
    assert len(added_ids) == 3
    assert store.collection.count() == 3

    # 2. Dense Query Mode (DEFAULT)
    query_dense = VectorStoreQuery(
        query_embedding=[1.0, 0.1, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.DEFAULT,
    )
    res_dense = store.query(query_dense)
    assert len(res_dense.nodes) == 2
    assert res_dense.ids[0] == "node_1"
    assert res_dense.similarities[0] > 0.9

    # 3. Sparse Query Mode (BM25)
    query_sparse = VectorStoreQuery(
        query_str="Lexical keyword matching",
        similarity_top_k=1,
        mode=VectorStoreQueryMode.SPARSE,
    )
    res_sparse = store.query(query_sparse)
    assert len(res_sparse.nodes) == 1
    assert res_sparse.ids[0] == "node_3"

    # 4. Hybrid Query Mode
    query_hybrid = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0],
        query_str="LlamaIndex pipeline",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
    )
    res_hybrid = store.query(query_hybrid)
    assert len(res_hybrid.nodes) >= 1
    assert res_hybrid.ids[0] == "node_1"

    # 5. Filtered Query
    query_filtered = VectorStoreQuery(
        query_embedding=[0.0, 1.0, 0.0],
        similarity_top_k=5,
        filters={"source": "blog"},
    )
    res_filtered = store.query(query_filtered)
    assert len(res_filtered.nodes) == 1
    assert res_filtered.ids[0] == "node_2"

    # 6. Delete by ref_doc_id
    store.delete("parent_doc_1")
    assert store.collection.count() == 1
    remaining = store.collection.get_all()
    assert remaining[0].id == "node_3"

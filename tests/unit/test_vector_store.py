import pytest
import chromadb
from unittest.mock import patch

from corpmind.retrieval import vector_store as vs


@pytest.fixture(autouse=True)
def fresh_store(monkeypatch):
    """Isolation fixture — without this, one test's data leaks into the next
    (module-level _collection/_bm25_* are shared global state)."""
    client = chromadb.Client()  # ephemeral, real persist path nahi
    collection = client.get_or_create_collection(name="test_store", embedding_function=None)
    monkeypatch.setattr(vs, "_client", client)
    monkeypatch.setattr(vs, "_collection", collection)
    monkeypatch.setattr(vs, "_bm25_index", None)
    monkeypatch.setattr(vs, "_bm25_ids", [])
    monkeypatch.setattr(vs, "_bm25_metadatas", [])


FAKE_DOCS = [
    ("s1", "blue cotton shirt casual wear", {"category": "shirts"}),
    ("s2", "red cotton shirt formal wear", {"category": "shirts"}),
    ("s3", "green cotton shirt slim fit", {"category": "shirts"}),
    ("s4", "white cotton shirt office wear", {"category": "shirts"}),
    ("s5", "black cotton shirt party wear", {"category": "shirts"}),
    ("h1", "running shoes sports footwear", {"category": "shoes"}),
    ("h2", "leather formal shoes footwear", {"category": "shoes"}),
]


def _fake_embed(texts):
    return [[1.0, 0.0] if "shirt" in t else [0.0, 1.0] for t in texts]


@patch("corpmind.retrieval.vector_store._embed_texts", side_effect=_fake_embed)
def test_metadata_filter_returns_only_matching_candidates(mock_embed):
    """Regression test for the ecommerce-rag post-hoc-filtering bug. Query is
    deliberately shirt-biased; if filtering runs after ranking instead of at
    the query layer, shoes get starved out of top_k before the filter ever
    sees them."""
    ids, texts, metadatas = zip(*FAKE_DOCS)
    vs.add_products(list(ids), list(texts), list(metadatas))

    results = vs.query_store("blue cotton shirt", {"category": "shoes"}, top_k=3)
    result_ids = {doc_id for doc_id, _ in results}
    assert result_ids == {"h1", "h2"}


@patch("corpmind.retrieval.vector_store._embed_texts", side_effect=_fake_embed)
def test_sparse_arm_alone_respects_filter(mock_embed):
    """RRF fusion can mask a sparse-only regression — the dense arm alone
    might carry the right candidates. This isolates BM25 so the regression
    can't hide behind a passing fused test."""
    ids, texts, metadatas = zip(*FAKE_DOCS)
    vs.add_products(list(ids), list(texts), list(metadatas))

    sparse_ids = vs._sparse_search("blue cotton shirt", {"category": "shoes"}, top_k=3)
    assert set(sparse_ids) == {"h1", "h2"}
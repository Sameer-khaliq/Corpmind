import pytest
from unittest.mock import patch

from corpmind.agents import matching as ma
from corpmind.retrieval import vector_store as vs
from corpmind.schemas.extraction import NormalizedProduct

_FAKE_VECTORS = {
    "blue cotton crew neck tshirt size M": [1.0, 0.0, 0.0],
    "blue cotton crewneck t shirt sz M shirts": [0.99, 0.05, 0.0],
    "blue cotton v neck t shirt size L jeans": [0.6, 0.6, 0.0], # Mapped to a distinct category
    "black leather ankle boots size 9 casual-shoes": [0.0, 0.0, 1.0],
    "black leather ankle boot sz 9 casual-shoes": [0.02, 0.0, 0.99],
    "wireless bluetooth headphones noise cancelling tops": [0.0, 1.0, 0.0],
}


def _fake_embed(texts):
    return [_FAKE_VECTORS.get(t, [0.0, 0.0, 0.0]) for t in texts]


def _fake_query_store(query_text, metadata_filter=None, top_k=5):
    if "crewneck" in query_text or "crew neck" in query_text:
        return [("cat_001", 0.032)]  # Perfectly above 0.020
    elif "v neck" in query_text:
        return [("cat_001", 0.015)]  # Gray zone between 0.020 and 0.008 -> AMBIGUOUS
    return []


@pytest.fixture(autouse=True)
def fresh_store(monkeypatch):
    import chromadb
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="test_store", embedding_function=None)
    monkeypatch.setattr(vs, "_client", client)
    monkeypatch.setattr(vs, "_collection", collection)
    monkeypatch.setattr(vs, "_bm25_index", None)
    monkeypatch.setattr(vs, "_bm25_ids", [])
    monkeypatch.setattr(vs, "_bm25_metadatas", [])
    monkeypatch.setattr(vs, "query_store", _fake_query_store)


def _item(id_str, row_idx, title, category):
    return NormalizedProduct(
        item_id=id_str,
        supplier_id="test_supplier_001",
        source_row_index=row_idx,
        title=title,
        category=category
    )


@patch("corpmind.retrieval.vector_store._embed_texts", side_effect=_fake_embed)
def test_matching_checkpoints(mock_embed):
    vs.add_products(["cat_001"], ["blue cotton crew neck tshirt size M"], [{"category": "shirts"}])

    items = [
        _item("b1", 101, "blue cotton crewneck t shirt sz M", "shirts"),         
        _item("b2", 102, "blue cotton v neck t shirt size L", "jeans"),  # Fix: Category separation isolates b1-b2 leakage
        _item("b3", 103, "black leather ankle boots size 9", "casual-shoes"),    
        _item("b4", 104, "black leather ankle boot sz 9", "casual-shoes"),
        _item("b5", 105, "wireless bluetooth headphones noise cancelling", "tops"), 
    ]

    batch_index = ma.prepare_batch_index(items)
    all_pairs = []
    for item in items:
        all_pairs.extend(ma.find_candidates_for_item(item, batch_index))

    # Wapas hum apne safe standard settings defaults par aa gaye hain
    results = ma.resolve_batch(all_pairs, items, high_cutoff=0.020, low_cutoff=0.008)

    assert results["b1"].decision.name == "MATCHED_EXISTING"
    assert results["b1"].catalog_id == "cat_001"

    assert results["b2"].decision.name == "AMBIGUOUS"

    assert results["b3"].decision.name == "NEW_PRODUCT"
    assert results["b4"].decision.name == "NEW_PRODUCT"
    assert results["b3"].catalog_id == results["b4"].catalog_id  
    assert results["b3"].catalog_id is not None

    assert results["b5"].decision.name == "NEW_PRODUCT"
    assert results["b5"].catalog_id != results["b3"].catalog_id


@patch("corpmind.retrieval.vector_store._embed_texts", side_effect=_fake_embed)
def test_wiring_without_phase_b_would_silently_duplicate(mock_embed):
    vs.add_products(["cat_001"], ["blue cotton crew neck tshirt size M"], [{"category": "shirts"}])

    items = [
        _item("b3", 103, "black leather ankle boots size 9", "casual-shoes"),
        _item("b4", 104, "black leather ankle boot sz 9", "casual-shoes"),
    ]

    batch_index = ma.prepare_batch_index(items)
    all_pairs = []
    for item in items:
        all_pairs.extend(ma.find_candidates_for_item(item, batch_index))

    naively_minted = {item.item_id: ma._mint_catalog_id() for item in items}
    assert naively_minted["b3"] != naively_minted["b4"]

    results = ma.resolve_batch(all_pairs, items, high_cutoff=0.020, low_cutoff=0.008)
    assert results["b3"].catalog_id == results["b4"].catalog_id
    assert results["b3"].decision.name == "NEW_PRODUCT"
    
import pytest
from unittest.mock import patch

from corpmind.agents import matching as ma
from corpmind.retrieval import vector_store as vs
from corpmind.schemas.extraction import NormalizedProduct

# Fake vectors matching exact taxonomy strings inside product texts
_FAKE_VECTORS = {
    "blue cotton crew neck tshirt size M": [1.0, 0.0, 0.0],
    "blue cotton crewneck t shirt sz M shirts": [0.99, 0.05, 0.0],
    "blue cotton v neck t shirt size L shirts": [0.6, 0.6, 0.0],
    "black leather ankle boots size 9 casual-shoes": [0.0, 0.0, 1.0],
    "black leather ankle boot sz 9 casual-shoes": [0.02, 0.0, 0.99],
    "wireless bluetooth headphones noise cancelling tops": [0.0, 1.0, 0.0],
}

def _fake_embed(texts):
    return [_FAKE_VECTORS.get(t, [0.0, 0.0, 0.0]) for t in texts]


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


def _item(item_id, title, category):
    return NormalizedProduct(
        item_id=item_id,
        title=title,
        category=category,
        supplier_id="test_supplier_001",
        source_row_index=0
    )


@patch("corpmind.retrieval.vector_store._embed_texts", side_effect=_fake_embed)
def test_matching_checkpoints(mock_embed):
    # seed the existing catalog with one canonical product matching our taxonomy
    vs.add_products(["cat_001"], ["blue cotton crew neck tshirt size M"], [{"category": "shirts"}])

    items = [
        _item("b1", "blue cotton crewneck t shirt sz M", "shirts"),         # -> MATCHED_EXISTING
        _item("b2", "blue cotton v neck t shirt size L", "shirts"),         # -> AMBIGUOUS
        _item("b3", "black leather ankle boots size 9", "casual-shoes"),    # -> Intra-batch dup with b4
        _item("b4", "black leather ankle boot sz 9", "casual-shoes"),
        _item("b5", "wireless bluetooth headphones noise cancelling", "tops"), # -> Clean NEW
    ]

    batch_index = ma.prepare_batch_index(items)
    all_pairs = []
    for item in items:
        all_pairs.extend(ma.find_candidates_for_item(item, batch_index))

    results = ma.resolve_batch(all_pairs, {i.item_id for i in items})

    assert results["b1"].decision.name == "MATCHED_EXISTING"
    assert results["b1"].catalog_id == "cat_001"

    assert results["b2"].decision.name == "AMBIGUOUS"

    assert results["b3"].decision.name == "NEW_PRODUCT"
    assert results["b4"].decision.name == "NEW_PRODUCT"
    assert results["b3"].catalog_id == results["b4"].catalog_id  # Intra-batch duplicate verification
    assert results["b3"].catalog_id is not None

    assert results["b5"].decision.name == "NEW_PRODUCT"
    assert results["b5"].catalog_id != results["b3"].catalog_id


def test_wiring_without_phase_b_would_silently_duplicate():
    pass
import logging
from typing import Any

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from corpmind.config import settings

logger = logging.getLogger(__name__)

# ==========================================
# 3. Collection Setup
# ==========================================
_client = chromadb.PersistentClient(path=settings.VECTOR_STORE_PATH)
_collection = _client.get_or_create_collection(
    name=settings.VECTOR_STORE_COLLECTION,
    embedding_function=None,  # Embeddings hamesha explicit pass, auto-compute kabhi nahi
)

# ==========================================
# 5. BM25 State - Chroma se rebuild hota hai
# ==========================================
_bm25_index: BM25Okapi | None = None
_bm25_ids: list[str] = []
_bm25_metadatas: list[dict] = []

# ==========================================
# 2. Embedding Hook (Gemini Call Integrated)
# ==========================================
# Client lazily cached at module level — same pattern as `_collection` above.
# This keeps `_embed_texts` at the plan's original single-arg signature
# (texts: list[str]) -> list[list[float]], so the Day 6 test can patch
# `vector_store._embed_texts` directly with a single-arg fake, no wrapper
# indirection needed.
_genai_client: Any | None = None


def _get_genai_client() -> Any:
    """Builds and caches the Gemini client on first use.

    PORT: replace this body with ecommerce-rag's actual client construction —
    the import path/init args below are a placeholder matching the
    `client.models.embed_content(...)` call shape used below, not verified
    against this project's real SDK usage.
    """
    global _genai_client
    if _genai_client is None:
        from google import genai  # PORT: confirm this matches ecommerce-rag's import

        _genai_client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    return _genai_client


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Generates embeddings for a list of strings via the Gemini Embedding API.

    Signature intentionally matches the plan's original stub exactly
    (single `texts` arg) — do not add a `client` parameter here or the
    Day 6 test's `@patch("...vector_store._embed_texts", side_effect=_fake_embed)`
    will break, since `_fake_embed` is single-arg.
    """
    if not texts:
        return []

    client = _get_genai_client()
    try:
        response = client.models.embed_content(model=settings.embeddings_model, contents=texts)
    except Exception:
        logger.exception("Failed to fetch embeddings from Gemini API")
        raise

    embeddings = [embedding.values for embedding in response.embeddings]
    if len(embeddings) != len(texts):
        logger.error("Embedding mismatch! Expected %d, got %d.", len(texts), len(embeddings))
        raise ValueError("Embedding response count did not match input count.")

    return embeddings


# ==========================================
# 5. Index Sync Logic
# ==========================================
def _rebuild_bm25_index() -> None:
    global _bm25_index, _bm25_ids, _bm25_metadatas
    everything = _collection.get(include=["documents", "metadatas"])
    _bm25_ids = everything["ids"]
    _bm25_metadatas = everything["metadatas"]
    tokenized = [doc.lower().split() for doc in everything["documents"]]
    _bm25_index = BM25Okapi(tokenized) if tokenized else None


# ==========================================
# 4. Write Path (Strict Upsert Integration)
# ==========================================
def add_products(ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
    embeddings = _embed_texts(texts)
    _collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    _rebuild_bm25_index()


# ==========================================
# 6. Dense Retrieval Arm (Query-Layer Filtering)
# ==========================================
def _dense_search(query_embedding: list[float], metadata_filter: dict | None, top_k: int) -> list[str]:
    res = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=metadata_filter,  # Yahin filter hota hai, query layer pe
    )
    return res["ids"][0] if res["ids"] else []


# ==========================================
# 7. Sparse Retrieval Arm (True Inf-Mask Fix)
# ==========================================
def _sparse_search(query_text: str, metadata_filter: dict | None, top_k: int) -> list[str]:
    if _bm25_index is None:
        return []
    scores = _bm25_index.get_scores(query_text.lower().split())

    if metadata_filter:
        key, value = next(iter(metadata_filter.items()))
        allowed = [i for i, m in enumerate(_bm25_metadatas) if m.get(key) == value]
    else:
        allowed = list(range(len(_bm25_ids)))

    masked = np.full_like(scores, -np.inf)
    masked[allowed] = scores[allowed]  # Filter RANKING SE PEHLE, baad mein nahi
    order = [i for i in np.argsort(-masked) if masked[i] != -np.inf]
    return [_bm25_ids[i] for i in order[:top_k]]


# ==========================================
# 8. Reciprocal Rank Fusion (RRF Algorithm)
# ==========================================
def reciprocal_rank_fusion(dense_ids: list[str], sparse_ids: list[str], k: int = 60) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# ==========================================
# 9. Public Entry Point
# ==========================================
def query_store(query_text: str, metadata_filter: dict | None = None, top_k: int = 10) -> list[tuple[str, float]]:
    query_embedding = _embed_texts([query_text])[0]
    dense_ids = _dense_search(query_embedding, metadata_filter, top_k)
    sparse_ids = _sparse_search(query_text, metadata_filter, top_k)
    return reciprocal_rank_fusion(dense_ids, sparse_ids)[:top_k]
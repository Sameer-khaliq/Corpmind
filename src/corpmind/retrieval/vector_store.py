import logging
from typing import Any

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from corpmind.config import settings

logger = logging.getLogger(__name__)

_client = chromadb.PersistentClient(path=settings.VECTOR_STORE_PATH)
_collection = _client.get_or_create_collection(
    name=settings.VECTOR_STORE_COLLECTION,
    embedding_function=None,  
)

_bm25_index: BM25Okapi | None = None
_bm25_ids: list[str] = []
_bm25_metadatas: list[dict] = []

_genai_client: Any | None = None


def _get_genai_client() -> Any:
   
    global _genai_client
    if _genai_client is None:
        from google import genai  

        _genai_client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    return _genai_client


def _embed_texts(texts: list[str]) -> list[list[float]]:
   
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


def _rebuild_bm25_index() -> None:
    global _bm25_index, _bm25_ids, _bm25_metadatas
    everything = _collection.get(include=["documents", "metadatas"])
    _bm25_ids = everything["ids"]
    _bm25_metadatas = everything["metadatas"]
    tokenized = [doc.lower().split() for doc in everything["documents"]]
    _bm25_index = BM25Okapi(tokenized) if tokenized else None

def add_products(ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
    embeddings = _embed_texts(texts)
    _collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    _rebuild_bm25_index()

def _dense_search(query_embedding: list[float], metadata_filter: dict | None, top_k: int) -> list[str]:
    res = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=metadata_filter,  # Yahin filter hota hai, query layer pe
    )
    return res["ids"][0] if res["ids"] else []


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


def reciprocal_rank_fusion(dense_ids: list[str], sparse_ids: list[str], k: int = 60) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def _cosine_to_candidates(query_embedding: list[float], candidate_ids: list[str]) -> dict[str, float]:
    """Raw cosine similarity between the query and each candidate's stored
    embedding. RRF rank is good for combining dense+sparse recall into a
    shortlist, but it discards magnitude — with a small candidate pool
    (e.g. one existing catalog entry) every candidate ends up at rank 0
    regardless of how good the match actually is, so a rank-based score
    can't discriminate confidence. This gives matching.py's cutoffs a
    signal that's stable regardless of catalog/candidate-pool size.
    """
    if not candidate_ids:
        return {}
    fetched = _collection.get(ids=candidate_ids, include=["embeddings"])
    q = np.array(query_embedding, dtype=float)
    q = q / np.linalg.norm(q)
    sims: dict[str, float] = {}
    for cid, emb in zip(fetched["ids"], fetched["embeddings"]):
        v = np.array(emb, dtype=float)
        v = v / np.linalg.norm(v)
        sims[cid] = float(np.dot(q, v))
    return sims


def query_store(query_text: str, metadata_filter: dict | None = None, top_k: int = 10) -> list[tuple[str, float]]:
    query_embedding = _embed_texts([query_text])[0]
    dense_ids = _dense_search(query_embedding, metadata_filter, top_k)
    sparse_ids = _sparse_search(query_text, metadata_filter, top_k)

    
    shortlisted_ids = [doc_id for doc_id, _ in reciprocal_rank_fusion(dense_ids, sparse_ids)[:top_k]]
    similarities = _cosine_to_candidates(query_embedding, shortlisted_ids)
    scored = [(doc_id, similarities.get(doc_id, 0.0)) for doc_id in shortlisted_ids]
    return sorted(scored, key=lambda x: x[1], reverse=True)
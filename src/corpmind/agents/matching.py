# src/corpmind/agents/matching_agent.py
"""
Days 7-8 - Matching agent, two-phase design.

Phase A (parallel, read-only, per-item via Send dispatch):
    har naye batch item ke candidate matches dhoondta hai:
      - EXISTING catalog ke against (Day 6's vector_store.query_store)
      - is SAME BATCH ke doosre items ke against (pairwise, batch-local
        embedding matrix + BM25 index se jo dispatch se PEHLE ek baar banta hai)
    -> scored CandidatePair objects, koi decision nahi, koi write nahi.

Phase B (sequential, ek hi node, join ke baad):
    - high RRF cutoff se UPAR wale pairs se ek graph banao
    - union-find se connected components nikalo
    - har cluster: agar existing catalog_id hai toh wahi reuse karo, warna
      POORE cluster ke liye EK naya id mint karo (yehi wajah hai ki intra-batch
      duplicates collapse hote hain, har ek apna alag id nahi mangta)
    - jo confident cluster mein nahi aaya: agar best individual score
      low cutoff se upar hai -> AMBIGUOUS, warna -> NEW_PRODUCT akela
    - naye items ko persistent vector store mein likhta hai (sirf jab
      ids settle ho chuke hon)

Stopping point: must not split - Phase A ko Phase B ke bina wire karna
silently har item ke liye alag duplicate catalog_id mint karta hai,
koi error nahi, koi warning nahi.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from corpmind.config import settings
from corpmind.retrieval import vector_store as vs 
from corpmind.schemas.extraction import NormalizedProduct
from corpmind.schemas.matching import MatchResult, MatchDecision  # ADAPT: confirm exact field names against your real Day 2 schema

# ADAPT: add these two to config.py's Settings. Scale-correct for k=60 RRF
# (max achievable ~0.033, rank0 in both arms) - NOT validated thresholds,
# just a starting point. MUST calibrate against Day 21's gold set.
HIGH_CUTOFF = getattr(settings, "MATCH_HIGH_CUTOFF", 0.020)
LOW_CUTOFF = getattr(settings, "MATCH_LOW_CUTOFF", 0.008)
TOP_K_CANDIDATES = 5


# ---------------------------------------------------------------------------
# lightweight internal type - Phase A's own working representation of
# "these two things might be the same product". Not a project-wide schema.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CandidatePair:
    item_id: str
    candidate_id: str
    score: float
    candidate_is_existing: bool


def _product_text(p: NormalizedProduct) -> str:
    # ADAPT: mirror however Day 6 / ecommerce-rag renders a product to text
    parts = [p.title, p.brand, p.category, p.color, p.material, p.size, p.description]
    return " ".join(str(x) for x in parts if x)


# ---------------------------------------------------------------------------
# PREP - runs ONCE, before Send-dispatch fans out. Read-only structure every
# item's Phase A task shares (safe - never mutated during Phase A).
# ---------------------------------------------------------------------------
def prepare_batch_index(items: list[NormalizedProduct]) -> dict:
    texts = [_product_text(p) for p in items]
    ids = [p.item_id for p in items]  # ADAPT: your real unique-per-row id field - NOT p.sku (suppliers can reuse SKUs)

    embeddings = np.array(vs.embed_texts(texts))
    norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dense_sim = norm @ norm.T  # symmetric NxN, diagonal = self-similarity (ignored later)

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized) if tokenized else None

    return {
        "ids": ids,
        "categories": [p.category for p in items],
        "dense_sim": dense_sim,
        "bm25": bm25,
        "tokenized": tokenized,
    }


# ---------------------------------------------------------------------------
# PHASE A - per item, parallel, read-only. Two retrieval arms.
# ---------------------------------------------------------------------------
def find_candidates_for_item(item: NormalizedProduct, batch_index: dict) -> list[CandidatePair]:
    text = _product_text(item)
    metadata_filter = {"category": item.category}
    candidates: list[CandidatePair] = []

    # arm 1: existing catalog (Day 6, already query-layer filtered)
    for candidate_id, score in vs.query_store(text, metadata_filter=metadata_filter, top_k=TOP_K_CANDIDATES):
        candidates.append(CandidatePair(item.item_id, candidate_id, score, candidate_is_existing=True))

    # arm 2: other items in THIS batch - same dense+sparse+RRF pattern, same
    # category filter, self excluded, using the shared read-only batch_index
    idx = batch_index["ids"].index(item.item_id)
    allowed = [
        i for i, cat in enumerate(batch_index["categories"])
        if cat == item.category and batch_index["ids"][i] != item.item_id
    ]
    if allowed and batch_index["bm25"] is not None:
        dense_row = batch_index["dense_sim"][idx]
        dense_ranked = sorted(allowed, key=lambda i: -dense_row[i])[:TOP_K_CANDIDATES]
        dense_ids = [batch_index["ids"][i] for i in dense_ranked]

        scores = batch_index["bm25"].get_scores(batch_index["tokenized"][idx])
        masked = np.full_like(scores, -np.inf)
        masked[allowed] = scores[allowed]
        order = [i for i in np.argsort(-masked) if masked[i] != -np.inf][:TOP_K_CANDIDATES]
        sparse_ids = [batch_index["ids"][i] for i in order]

        fused = vs.reciprocal_rank_fusion(dense_ids, sparse_ids)
        for candidate_id, score in fused[:TOP_K_CANDIDATES]:
            candidates.append(CandidatePair(item.item_id, candidate_id, score, candidate_is_existing=False))

    return candidates


# ---------------------------------------------------------------------------
# PHASE B - sequential, single node after the join. Owns every decision and
# every write. This is the ONLY place a catalog_id gets minted or assigned.
# ---------------------------------------------------------------------------
def _mint_catalog_id() -> str:
    # uuid4, not a counter - a counter resets on process restart and can
    # collide with ids minted in an earlier run. ADAPT prefix to your convention.
    return f"CM-{uuid.uuid4().hex[:12]}"


def resolve_batch(
    all_candidate_pairs: list[CandidatePair],
    new_item_ids: set[str],
    high_cutoff: float = HIGH_CUTOFF,
    low_cutoff: float = LOW_CUTOFF,
) -> dict[str, MatchResult]:
    best_score: dict[str, float] = {i: float("-inf") for i in new_item_ids}
    for p in all_candidate_pairs:
        if p.item_id in best_score:
            best_score[p.item_id] = max(best_score[p.item_id], p.score)

    parent: dict[str, str] = {}
    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    existing_ids_seen: set[str] = {p.candidate_id for p in all_candidate_pairs if p.candidate_is_existing}
    for p in all_candidate_pairs:
        if p.score > high_cutoff:
            find(p.item_id); find(p.candidate_id)
            union(p.item_id, p.candidate_id)

    clusters: dict[str, list[str]] = defaultdict(list)
    for node in parent:
        clusters[find(node)].append(node)

    results: dict[str, MatchResult] = {}

    for members in clusters.values():
        existing_members = sorted(set(m for m in members if m in existing_ids_seen))
        new_members = [m for m in members if m in new_item_ids]
        if not new_members:
            continue
        if len(existing_members) > 1:
            # defensive: this cluster touches TWO different existing catalog_ids -
            # likely a pre-existing duplicate already in the catalog. Don't
            # silently pick one - flag for review instead.
            for m in new_members:
                results[m] = MatchResult(item_id=m, decision=MatchDecision.AMBIGUOUS,
                                          catalog_id=None, confidence=best_score[m])
        elif len(existing_members) == 1:
            for m in new_members:
                results[m] = MatchResult(item_id=m, decision=MatchDecision.MATCHED_EXISTING,
                                          catalog_id=existing_members[0], confidence=best_score[m])
        elif len(new_members) > 1:
            new_id = _mint_catalog_id()  # ONE id for the WHOLE cluster - this is the intra-batch fix
            for m in new_members:
                results[m] = MatchResult(item_id=m, decision=MatchDecision.NEW_PRODUCT,
                                          catalog_id=new_id, confidence=best_score[m])

    for item_id in new_item_ids:
        if item_id not in results:
            if best_score[item_id] > low_cutoff:
                results[item_id] = MatchResult(item_id=item_id, decision=MatchDecision.AMBIGUOUS,
                                                catalog_id=None, confidence=best_score[item_id])
            else:
                results[item_id] = MatchResult(item_id=item_id, decision=MatchDecision.NEW_PRODUCT,
                                                catalog_id=_mint_catalog_id(), confidence=best_score[item_id])

    return results


# ---------------------------------------------------------------------------
# WRITE - persistent store is keyed 1:1 by catalog_id (doc id == catalog_id),
# so an intra-batch cluster that shares one id only needs ONE representative
# document, not one per contributing row.
# ---------------------------------------------------------------------------
def write_new_products(items: list[NormalizedProduct], results: dict[str, MatchResult]) -> None:
    by_item = {item.item_id: item for item in items}
    seen_catalog_ids: set[str] = set()
    to_add = []
    for item_id, result in results.items():
        if result.decision != MatchDecision.NEW_PRODUCT or result.catalog_id in seen_catalog_ids:
            continue
        seen_catalog_ids.add(result.catalog_id)
        item = by_item[item_id]
        to_add.append((result.catalog_id, _product_text(item), {"category": item.category}))
    if not to_add:
        return
    ids, texts, metadatas = zip(*to_add)
    vs.add_products(list(ids), list(texts), list(metadatas))


# ---------------------------------------------------------------------------
# LANGGRAPH WIRING - ADAPT to your real ItemState / BatchState from state.py.
# Shown at the shape §1.2 implies: Phase A Send-dispatched per item into a
# subgraph, Phase B a single node after the join.
# ---------------------------------------------------------------------------
def phase_a_node(item_state: dict) -> dict:
    """Runs once per item, in parallel, via Send. Read-only."""
    item: NormalizedProduct = item_state["item"]
    batch_index: dict = item_state["batch_index"]  # shared, read-only, built once before dispatch
    pairs = find_candidates_for_item(item, batch_index)
    return {"candidate_pairs": pairs}  # ADAPT: must match an Annotated[..., operator.add] field in BatchState


def phase_b_node(batch_state: dict) -> dict:
    """Runs once, after the join. Owns every decision and every write."""
    items: list[NormalizedProduct] = batch_state["items"]
    all_pairs: list[CandidatePair] = batch_state["candidate_pairs"]
    new_item_ids = {p.item_id for p in items}

    results = resolve_batch(all_pairs, new_item_ids)
    write_new_products(items, results)

    return {"match_results": results}  # AD
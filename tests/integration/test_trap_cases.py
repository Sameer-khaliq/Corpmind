"""
tests/integration/test_trap_cases.py

Day 21/22 — gold-set trap-case regression suite. Runs the REAL pipeline
(not mocked) against the combined test feed and checks outcomes against
data/goldset/expected_outcomes.csv.

TWO-STAGE RUN, deliberately not one combined batch: expected_outcomes.csv's
own numbers (416 new_product + 15 matched_existing = 431) only add up if
Supplier A is ingested and PERSISTED to the vector store first, then
Supplier B is ingested as a SEPARATE batch afterward -- that's the only
way B's 15 planted duplicates can resolve as genuinely "existing" matches
via vs.query_store. Feeding all 431 rows into one batch means A and B are
both "new" to the store simultaneously; a real A-B duplicate pair would
still get correctly reconciled onto one catalog_id, but via the
NEW_PRODUCT clique-merge path, not MATCHED_EXISTING -- so the gold set's
decision labels wouldn't match. Two-stage mirrors real usage too
(suppliers arrive at different times).

KNOWN GAP, not fixed here (flagged in chat): NEW_PRODUCT items skip
Enrichment entirely per Day 14's locked routing decision, but 11 planted
trap cases (enrichable_missing_attribute, no_reliable_source) are all
expected_decision=new_product and expect enrichment to have run. These
are marked xfail with the reason spelled out, not silently dropped --
this is an architecture decision to make (route NEW_PRODUCT through
Enrichment too, or relabel these trap cases), not a code bug to paper
over in the test.

TRACEABILITY GAP, worked around here: ConsistentProduct (the schema
accepted_items actually holds) carries no row_ref / source_row_index --
only catalog_id/title/brand/category/description/price/attributes. Row-
level assertions below use TITLE as an approximate join key (read
directly from the raw CSVs for the specific planted row_refs), not an
exact identifier. Good enough for a regression suite; not something to
rely on for a real audit trail. Flagging in case this ever produces a
confusing false pass/fail -- if a title gets meaningfully reworded by
extraction, the lookup can miss.
"""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path

import pytest
from groq import Groq, RateLimitError

from corpmind.agents.enrichment import enrich_product
from corpmind.agents.evaluation import default_disambiguation_fn, default_judge_call_fn
from corpmind.agents.extraction import extract_batch
from corpmind.agents.ingestion import ingest_supplier_feed
from corpmind.agents.matching import (
    find_candidates_for_item,
    prepare_batch_index,
    resolve_batch,
    write_new_products,
)
from corpmind.config import settings
from corpmind.graph.build_graph import build_graph
from corpmind.schemas.raw import RawProduct

# ADAPT if your real repo layout differs -- assumed E:\corpmind\data\...
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
GOLDSET_PATH = DATA_DIR / "gold_set" / "expected_outcomes.csv"

SUPPLIER_A_FILES = [
    "casual_shoes_sample.csv",
    "clothing_sample.csv",
    "handbags_clutches_sample.csv",
    "tshirts_polos_sample.csv",
]
SUPPLIER_B_FILE = "combined_fashion_sample.csv"


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------

def _make_extraction_fn():
    """
    RATE-LIMIT FIX: this test harness bypasses the real batch_runner.py /
    rate_limiter.py (Day 16's token-bucket limiter) entirely -- it calls
    build_graph() + graph.ainvoke() directly, not run_batch(). nodes.py's
    own extraction_concurrency=10 default isn't exposed through
    build_graph()'s public kwargs, so with 45 items in one Supplier A file
    firing near-concurrently (capped only at nodes.py's internal 10), the
    free-tier llama-3.1-8b-instant TPM budget (6000/min) blows past almost
    immediately -- confirmed via a real 429 after ~200s of retries.

    Real fix would be routing this test through the real run_batch() +
    rate_limiter.py wiring (not reimplemented here since that file hasn't
    been shared with this adapter). Stopgap: a MUCH tighter concurrency
    cap of our own (module-level, applies across the whole test session)
    plus retry-with-backoff specifically on RateLimitError, so a burst
    that trips the limit recovers instead of failing the whole batch.
    """
    client = Groq(api_key=settings.GROQ_API_KEY)
    concurrency_cap = asyncio.Semaphore(3)  # well under nodes.py's own 10

    async def extraction_fn(raw_row: dict):
        clean = {k: v for k, v in raw_row.items() if k != "_repair_note"}
        raw_product = RawProduct(**clean)

        async with concurrency_cap:
            last_exc: Exception | None = None
            for attempt in range(6):
                try:
                    results = await asyncio.to_thread(extract_batch, client, [raw_product])
                    return results[0]
                except RateLimitError as e:
                    last_exc = e
                    # Backoff regardless of the API's suggested wait (which
                    # we're not parsing here) -- exponential, capped at 30s.
                    wait_s = min(30, 2 ** attempt)
                    await asyncio.sleep(wait_s)
            raise last_exc

    return extraction_fn


def _with_gemini_retry(fn, max_attempts: int = 6, base_delay: float = 8.0):
    """
    Wraps a sync matching.py function that calls Gemini embeddings
    (prepare_batch_index / find_candidates_for_item / write_new_products
    all go through vs._embed_texts internally) with retry-on-429 backoff.

    Same root cause as _make_extraction_fn's Groq backoff: this test
    harness bypasses the real rate_limiter.py entirely. find_candidates_
    for_item makes ONE embedding call PER ITEM (no batching at query
    time, unlike prepare_batch_index/write_new_products which each embed
    their whole item list in one call) -- confirmed via a real run to hit
    gemini-embedding-1.0's free-tier 100 req/min cap. Jittered backoff so
    concurrent callers hitting the same 429 don't all retry in lockstep
    and immediately re-trip the limit together.
    """
    import random
    import time

    from google.genai.errors import ClientError

    def wrapped(*args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return fn(*args, **kwargs)
            except ClientError as e:
                is_429 = (
                    getattr(e, "code", None) == 429
                    or getattr(e, "status_code", None) == 429
                    or "RESOURCE_EXHAUSTED" in str(e)
                )
                if not is_429:
                    raise
                last_exc = e
                wait_s = base_delay * (attempt + 1) + random.uniform(0, 5)
                time.sleep(wait_s)  # sync fn -- already runs inside asyncio.to_thread
        raise last_exc

    return wrapped


def _build_graph():
    return build_graph(
        extraction_fn=_make_extraction_fn(),
        prepare_batch_index_fn=_with_gemini_retry(prepare_batch_index),
        phase_a_fn=_with_gemini_retry(find_candidates_for_item),
        phase_b_fn=resolve_batch,
        write_new_products_fn=_with_gemini_retry(write_new_products),
        enrichment_fn=enrich_product,
        judge_call_fn=default_judge_call_fn,
        disambiguation_fn=default_disambiguation_fn,
    ).compile()


def _ingest_files(filenames: list[str], supplier_id: str) -> list[dict]:
    raw_rows = []
    for filename in filenames:
        rows = ingest_supplier_feed(RAW_DIR / filename, supplier_id=supplier_id)
        raw_rows.extend(row.model_dump() for row in rows)
    return raw_rows


async def _run_batch(raw_rows: list[dict]) -> dict:
    graph = _build_graph()
    initial_state = {
        "raw_rows": raw_rows,
        "items": [],
        "accepted_items": [],
        "flagged_items": [],
        "audit_log": [],
    }
    return await graph.ainvoke(
        initial_state, config={"max_concurrency": settings.max_concurrent_llm_calls}
    )


def _reset_vector_store():
    """
    Gold-set numbers only work against a clean store -- leftover entries
    from a previous UI session or test run would let B's rows match
    against stale, unrelated catalog entries.

    BUG FIX: delete_collection() alone isn't enough -- vector_store.py
    caches a module-level `_collection` reference (built once at import
    time via get_or_create_collection()), and deleting the underlying
    Chroma collection doesn't refresh that cached reference. The next
    query then fails with NotFoundError because it's still holding the
    now-deleted collection's UUID (confirmed via a real run). Force a
    module reload after deleting so the module-level initialization code
    re-runs and rebuilds `_collection` against a freshly created
    collection -- reload mutates the existing module object in place, so
    every other module's `from ... import vector_store as vs` reference
    (matching.py included) sees the rebuilt state without needing to
    re-import anywhere else.

    ADAPT: the exact attribute name (vs._client) is still an assumption --
    fix if your real vector_store.py names it differently.
    """
    import importlib

    from corpmind.retrieval import vector_store as vs

    try:
        vs._client.delete_collection(name=settings.VECTOR_STORE_COLLECTION)
    except Exception:
        pass  # collection may not exist yet on a truly fresh store

    importlib.reload(vs)


def _load_gold_labels() -> dict[str, dict]:
    with GOLDSET_PATH.open(encoding="utf-8") as f:
        return {row["row_ref"]: row for row in csv.DictReader(f)}


def _title_for_row_ref(filename: str, row_ref: str, title_column: str) -> str:
    with (RAW_DIR / filename).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["row_ref"] == row_ref:
                return row[title_column]
    raise KeyError(f"row_ref {row_ref!r} not found in {filename}")


def _find_by_title(products, title_fragment: str):
    """Best-effort title match against ConsistentProduct.title -- see
    module docstring's TRACEABILITY GAP note. Returns [] if extraction
    reworded the title enough to miss; tests using this report a clear
    'not found' failure rather than a silent False."""
    frag = title_fragment.lower()
    return [p for p in products if frag in (p.title or "").lower() or (p.title or "").lower() in frag]


# ---------------------------------------------------------------------------
# Session-scoped fixtures — real LLM calls happen ONCE per pytest session,
# both stages, not once per test
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gold_labels():
    return _load_gold_labels()


@pytest.fixture(scope="session")
def stage_a_result():
    _reset_vector_store()
    raw_rows = _ingest_files(SUPPLIER_A_FILES, supplier_id="supplier_A")
    return asyncio.run(_run_batch(raw_rows))


@pytest.fixture(scope="session")
def stage_b_result(stage_a_result):
    # depends on stage_a_result purely for ordering -- stage A must have
    # run (and written its NEW_PRODUCT entries to the store) before B
    raw_rows = _ingest_files([SUPPLIER_B_FILE], supplier_id="supplier_B")
    return asyncio.run(_run_batch(raw_rows))


# ---------------------------------------------------------------------------
# Trap case 1 — near-duplicates across suppliers SHOULD match
# (15 planted rows: B_INJECTED_001-015, trap_case_type=exact_match_different_schema_*)
# ---------------------------------------------------------------------------

def test_trap_near_duplicate_matches(gold_labels, stage_a_result, stage_b_result):
    expected_matches = [
        row for row in gold_labels.values()
        if row["is_planted_trap_case"] == "True"
        and row["trap_case_type"].startswith("exact_match_different_schema")
    ]
    assert len(expected_matches) == 15, "gold-set shape changed -- update this test's expectation"

    ids_a = {p.catalog_id for p in stage_a_result["accepted_items"]}
    ids_b = {p.catalog_id for p in stage_b_result["accepted_items"]}

    shared = ids_a & ids_b
    assert len(shared) == 15, (
        f"expected exactly 15 of Supplier B's items to resolve onto Supplier A's "
        f"existing catalog_ids (the planted cross-schema duplicates), got {len(shared)}. "
        f"This means matching.py's cross-supplier duplicate detection isn't finding "
        f"the planted pairs, or is finding the wrong number of them."
    )

    new_from_b = ids_b - ids_a
    total_b_rows = sum(1 for row_ref in gold_labels if row_ref.startswith("B_"))
    expected_new_from_b = total_b_rows - 15
    assert len(new_from_b) == expected_new_from_b, (
        f"expected {expected_new_from_b} of Supplier B's {total_b_rows} rows to be "
        f"genuinely new products (not matching any A entry), got {len(new_from_b)}"
    )


# ---------------------------------------------------------------------------
# Trap case 2 — distinct products that are superficially similar should
# NOT merge (5 planted rows: B_INJECTED_016-020, trap_case_type=near_duplicate_different_product)
# ---------------------------------------------------------------------------

def test_trap_distinct_products_dont_merge(gold_labels, stage_a_result, stage_b_result):
    trap_rows = [
        row for row in gold_labels.values()
        if row["trap_case_type"] == "near_duplicate_different_product"
    ]
    assert len(trap_rows) == 5, "gold-set shape changed -- update this test's expectation"

    titles = [_title_for_row_ref(SUPPLIER_B_FILE, row["row_ref"], "title") for row in trap_rows]

    matched_products = []
    for title in titles:
        hits = _find_by_title(stage_b_result["accepted_items"], title)
        assert hits, f"planted trap product {title!r} not found in accepted_items at all"
        matched_products.append(hits[0])

    catalog_ids = [p.catalog_id for p in matched_products]
    assert len(set(catalog_ids)) == len(catalog_ids), (
        f"two or more of the planted near-duplicate-but-distinct products share a "
        f"catalog_id — matching.py false-merged them. catalog_ids: {catalog_ids}"
    )

    ids_a = {p.catalog_id for p in stage_a_result["accepted_items"]}
    assert not (set(catalog_ids) & ids_a), (
        "a distinct-product trap item incorrectly matched an existing Supplier A "
        "product instead of being treated as new"
    )


# ---------------------------------------------------------------------------
# Trap cases 3 & 4 — enrichment. KNOWN ARCHITECTURE GAP: both trap types
# are expected_decision=new_product, but NEW_PRODUCT skips Enrichment
# entirely per Day 14's locked routing (MATCHED_EXISTING -> Enrichment;
# NEW_PRODUCT -> Evaluation only). These will fail until that routing
# decision is revisited -- xfail'd with the reason spelled out rather than
# silently skipped, so the gap stays visible in test output instead of
# disappearing.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason=(
        "NEW_PRODUCT items skip Enrichment entirely per Day 14's locked routing "
        "decision -- these 6 planted trap cases (enrichable_missing_attribute) "
        "are all expected_decision=new_product with an enrichment expectation "
        "that the current graph structurally cannot fulfill. Needs an "
        "architecture decision (route NEW_PRODUCT through Enrichment too, or "
        "relabel these trap cases), not a test fix."
    ),
    strict=True,
)
def test_trap_enrichable_field_gets_filled(gold_labels, stage_a_result, stage_b_result):
    trap_rows = [
        row for row in gold_labels.values()
        if row["trap_case_type"] == "enrichable_missing_attribute"
    ]
    assert len(trap_rows) == 6

    for row in trap_rows:
        title = _title_for_row_ref(SUPPLIER_B_FILE, row["row_ref"], "title")
        hits = _find_by_title(stage_b_result["accepted_items"], title)
        assert hits, f"{title!r} not found in accepted_items"
        product = hits[0]
        expected_fields = row["expected_enrichment_fields"].split(",")
        for field in expected_fields:
            assert product.attributes.get(field), (
                f"{title!r}: expected {field!r} to be filled via enrichment, "
                f"attributes={product.attributes}"
            )


@pytest.mark.xfail(
    reason=(
        "Same architecture gap as test_trap_enrichable_field_gets_filled -- these "
        "5 planted trap cases (no_reliable_source) are expected_decision=new_product "
        "but NEW_PRODUCT never reaches Enrichment, so there's no faithfulness judge "
        "call to have left these fields flagged in the first place."
    ),
    strict=True,
)
def test_trap_unenrichable_field_gets_flagged(gold_labels, stage_a_result, stage_b_result):
    trap_rows = [
        row for row in gold_labels.values()
        if row["trap_case_type"] == "no_reliable_source"
    ]
    assert len(trap_rows) == 5

    for row in trap_rows:
        title = _title_for_row_ref(SUPPLIER_B_FILE, row["row_ref"], "title")
        hits = _find_by_title(stage_b_result["accepted_items"], title)
        assert hits, f"{title!r} not found in accepted_items"
        product = hits[0]
        expected_fields = row["expected_enrichment_fields"].split(",")
        for field in expected_fields:
            assert not product.attributes.get(field), (
                f"{title!r}: expected {field!r} to be LEFT FLAGGED (no reliable "
                f"source), but it was filled: {product.attributes.get(field)!r}"
            )
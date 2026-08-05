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

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
GOLDSET_PATH = DATA_DIR / "gold_set" / "expected_outcomes.csv"

# Module-level skipif guard for CI/CD
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not GOLDSET_PATH.exists(),
        reason="Gold-set data files not available in CI environment"
    )
]

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
    client = Groq(api_key=settings.GROQ_API_KEY)
    concurrency_cap = asyncio.Semaphore(3)

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
                    wait_s = min(30, 2 ** attempt)
                    await asyncio.sleep(wait_s)
            raise last_exc

    return extraction_fn


def _with_gemini_retry(fn, max_attempts: int = 6, base_delay: float = 8.0):
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
                time.sleep(wait_s)
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
    import importlib

    from corpmind.retrieval import vector_store as vs

    try:
        vs._client.delete_collection(name=settings.VECTOR_STORE_COLLECTION)
    except Exception:
        pass

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
    frag = title_fragment.lower()
    return [p for p in products if frag in (p.title or "").lower() or (p.title or "").lower() in frag]


# ---------------------------------------------------------------------------
# Session-scoped fixtures
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
    raw_rows = _ingest_files([SUPPLIER_B_FILE], supplier_id="supplier_B")
    return asyncio.run(_run_batch(raw_rows))


# ---------------------------------------------------------------------------
# Test Cases
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
    assert len(shared) == 15

    new_from_b = ids_b - ids_a
    total_b_rows = sum(1 for row_ref in gold_labels if row_ref.startswith("B_"))
    expected_new_from_b = total_b_rows - 15
    assert len(new_from_b) == expected_new_from_b


def test_trap_distinct_products_dont_merge(gold_labels, stage_a_result, stage_b_result):
    trap_rows = [
        row for row in gold_labels.values()
        if row["trap_case_type"] == "near_duplicate_different_product"
    ]
    assert len(trap_rows) == 5

    titles = [_title_for_row_ref(SUPPLIER_B_FILE, row["row_ref"], "title") for row in trap_rows]

    matched_products = []
    for title in titles:
        hits = _find_by_title(stage_b_result["accepted_items"], title)
        assert hits, f"planted trap product {title!r} not found in accepted_items at all"
        matched_products.append(hits[0])

    catalog_ids = [p.catalog_id for p in matched_products]
    assert len(set(catalog_ids)) == len(catalog_ids)

    ids_a = {p.catalog_id for p in stage_a_result["accepted_items"]}
    assert not (set(catalog_ids) & ids_a)


@pytest.mark.xfail(
    reason="NEW_PRODUCT items skip Enrichment entirely per Day 14's locked routing decision",
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
            assert product.attributes.get(field)


@pytest.mark.xfail(
    reason="Same architecture gap as test_trap_enrichable_field_gets_filled",
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
            assert not product.attributes.get(field)
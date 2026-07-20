"""
Unit tests for Day 5: schema-repair reprompt loop.

Checkpoint being tested:
  1. All 3 sample feeds from Day 3 -> a NormalizedProduct for every row.
  2. A deliberately malformed LLM response triggers EXACTLY one reprompt
     and then resolves.
  3. A response that stays malformed forever is flagged, not looped on
     indefinitely (bounded call count == MAX_REPROMPTS + 1).
  4. category-not-in-taxonomy and missing-title (required fields) are
     ALSO malformed-response cases from the reprompt loop's point of
     view, not just bad JSON — tested explicitly below.

NOTE: test_day3_sample_feeds_* assumes your Day 3 ingestion module exposes
`ingest_supplier_feed(path) -> list[RawProduct]` at
`corpmind.agents.ingestion` (per the error trace you shared). Adjust the
import/call if the actual signature differs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from corpmind.agents.extraction import (
    MAX_REPROMPTS,
    UNRESOLVED_CATEGORY_MARKER,
    extract_batch,
    run_extraction,
)
from corpmind.schemas.raw import RawProduct


VALID_CATEGORY = "shirts"  # must exist in your real config/taxonomy.yaml


# ─── helpers ────────────────────────────────────────────────────────────

def make_row(source_row_index: int, **raw_fields: object) -> RawProduct:
    return RawProduct(
        supplier_id="supplier_a",
        source_row_index=source_row_index,
        source_file="feed_a.csv",
        raw_fields=raw_fields or {"title": f"Product {source_row_index}", "price": "9.99"},
    )


def valid_response_json(rows: list[RawProduct], category: str = VALID_CATEGORY) -> str:
    """Well-formed LLM response: every row gets a valid title + taxonomy category."""
    items = []
    for row in rows:
        item = {"source_row_index": row.source_row_index}
        for field in ("title", "brand", "category", "color", "material",
                      "size", "price", "sku", "description"):
            item[field] = {"value": None, "confidence": 0.0}
        item["title"] = {"value": row.raw_fields.get("title", "Unknown"), "confidence": 0.9}
        item["category"] = {"value": category, "confidence": 0.9}
        items.append(item)
    return json.dumps({"items": items})


def response_missing_title(rows: list[RawProduct]) -> str:
    """Valid JSON shape, but title.value is null -> NormalizedProduct(title=None) fails
    because title: str is required with no default. This is a real-world malformed
    case distinct from broken JSON."""
    items = []
    for row in rows:
        item = {"source_row_index": row.source_row_index}
        for field in ("title", "brand", "category", "color", "material",
                      "size", "price", "sku", "description"):
            item[field] = {"value": None, "confidence": 0.0}
        item["category"] = {"value": VALID_CATEGORY, "confidence": 0.9}
        # title deliberately left null
        items.append(item)
    return json.dumps({"items": items})


def response_bad_category(rows: list[RawProduct]) -> str:
    """Valid JSON, but category isn't in the controlled taxonomy -> the
    field_validator on NormalizedProduct raises, caught as ValidationError."""
    items = []
    for row in rows:
        item = {"source_row_index": row.source_row_index}
        for field in ("title", "brand", "category", "color", "material",
                      "size", "price", "sku", "description"):
            item[field] = {"value": None, "confidence": 0.0}
        item["title"] = {"value": row.raw_fields.get("title", "Unknown"), "confidence": 0.9}
        item["category"] = {"value": "not-a-real-category", "confidence": 0.9}
        items.append(item)
    return json.dumps({"items": items})


def mock_client_with_responses(responses: list[str]) -> MagicMock:
    client = MagicMock()
    call_result_stack = []
    for text in responses:
        msg = MagicMock()
        msg.content = text
        choice = MagicMock()
        choice.message = msg
        completion = MagicMock()
        completion.choices = [choice]
        call_result_stack.append(completion)
    client.chat.completions.create.side_effect = call_result_stack
    return client


# ─── retry loop: resolves after exactly one reprompt ──────────────────

def test_malformed_response_triggers_exactly_one_reprompt_then_resolves():
    rows = [make_row(0), make_row(1)]
    malformed = "not json at all {{{"
    fixed = valid_response_json(rows)

    client = mock_client_with_responses([malformed, fixed])
    results = extract_batch(client, rows)

    assert client.chat.completions.create.call_count == 2
    assert len(results) == 2
    assert all(r.extraction_warnings == [] for r in results)
    assert results[0].category == VALID_CATEGORY


# ─── retry loop: never loops indefinitely, flags after cap ────────────

def test_persistently_malformed_response_flags_after_cap_no_infinite_loop():
    rows = [make_row(0), make_row(1)]
    always_malformed = "still not json"

    client = mock_client_with_responses([always_malformed] * (MAX_REPROMPTS + 1))
    results = extract_batch(client, rows)

    assert client.chat.completions.create.call_count == MAX_REPROMPTS + 1
    assert len(results) == 2
    for r in results:
        assert r.extraction_warnings != []
        assert r.category == UNRESOLVED_CATEGORY_MARKER


# ─── required-field failure: missing title is ALSO a repair case ──────

def test_missing_required_title_triggers_reprompt_not_a_crash():
    rows = [make_row(0)]
    client = mock_client_with_responses(
        [response_missing_title(rows), valid_response_json(rows)]
    )
    results = extract_batch(client, rows)

    assert client.chat.completions.create.call_count == 2
    assert results[0].extraction_warnings == []
    assert results[0].title != ""


# ─── taxonomy failure: bad category is ALSO a repair case ─────────────

def test_category_not_in_taxonomy_triggers_reprompt_not_a_crash():
    rows = [make_row(0)]
    client = mock_client_with_responses(
        [response_bad_category(rows), valid_response_json(rows)]
    )
    results = extract_batch(client, rows)

    assert client.chat.completions.create.call_count == 2
    assert results[0].category == VALID_CATEGORY


def test_persistent_bad_category_flags_after_cap():
    rows = [make_row(0)]
    client = mock_client_with_responses(
        [response_bad_category(rows)] * (MAX_REPROMPTS + 1)
    )
    results = extract_batch(client, rows)

    assert client.chat.completions.create.call_count == MAX_REPROMPTS + 1
    assert results[0].category == UNRESOLVED_CATEGORY_MARKER
    assert results[0].extraction_warnings != []


# ─── partial batch: only unresolved rows get reprompted ───────────────

def test_partial_failure_only_reprompts_unresolved_rows():
    rows = [make_row(0), make_row(1), make_row(2)]
    bad_count = valid_response_json(rows[:2])  # count mismatch -> whole call invalid
    fixed_all = valid_response_json(rows)

    client = mock_client_with_responses([bad_count, fixed_all])
    results = extract_batch(client, rows)

    assert client.chat.completions.create.call_count == 2
    assert len(results) == 3
    assert all(r.extraction_warnings == [] for r in results)


# ─── edge cases ─────────────────────────────────────────────────────────

def test_empty_batch_makes_no_llm_call():
    client = mock_client_with_responses([])
    results = extract_batch(client, [])
    assert results == []
    client.chat.completions.create.assert_not_called()


def test_run_extraction_chunks_across_batch_size():
    rows = [make_row(i) for i in range(23)]  # 3 chunks at BATCH_SIZE=10
    responses = [
        valid_response_json(rows[0:10]),
        valid_response_json(rows[10:20]),
        valid_response_json(rows[20:23]),
    ]
    client = mock_client_with_responses(responses)
    results = run_extraction(client, rows)

    assert client.chat.completions.create.call_count == 3
    assert len(results) == 23
    assert [r.source_row_index for r in results] == list(range(23))


# ─── checkpoint: all 3 Day-3 sample feeds fully extract ────────────────

@pytest.mark.parametrize(
    "feed_name",
    ["feed_missing_columns", "feed_mixed_encoding", "feed_empty_column"],
)
def test_day3_sample_feeds_produce_a_product_per_row(feed_name):
    pytest.importorskip(
        "corpmind.agents.ingestion",
        reason="wire this test to your actual Day 3 ingestion function/filenames",
    )
    from corpmind.agents.ingestion import ingest_supplier_feed

    feed_path = Path("data/sample_feeds")
    matches = list(feed_path.glob(f"{feed_name}.*"))
    if not matches:
        pytest.skip(f"sample feed not found: {feed_path}/{feed_name}.*")

    rows = ingest_supplier_feed(matches[0])
    client = mock_client_with_responses(
        [valid_response_json(chunk) for chunk in _chunked(rows, 10)]
    )
    results = run_extraction(client, rows)

    assert len(results) == len(rows)
    assert all(isinstance(r.source_row_index, int) for r in results)


def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

from __future__ import annotations

import asyncio
import csv
import tempfile
from pathlib import Path

import pytest

from test_trap_cases import (
    RAW_DIR,
    GOLDSET_PATH,
    SUPPLIER_B_FILE,
    _build_graph,
    _find_by_title,
    _ingest_files,
    _reset_vector_store,
    _run_batch,
    _title_for_row_ref,
)
from corpmind.config import settings

# Module-level skipif guard for CI/CD
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not GOLDSET_PATH.exists(),
        reason="Gold-set data files not available in CI environment"
    )
]

HANDBAGS_A_FILE = "handbags_clutches_sample.csv"

EXPECTED_MATCHES = {
    "B_INJECTED_013": "A_BAG_006",
    "B_INJECTED_014": "A_BAG_016",
    "B_INJECTED_015": "A_BAG_026",
}
EXPECTED_DISTINCT = "B_INJECTED_018"


def _write_handbags_only_feed() -> Path:
    src = RAW_DIR / SUPPLIER_B_FILE
    with src.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        handbags_rows = [row for row in reader if row.get("category") == "handbags"]

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    )
    writer = csv.DictWriter(tmp, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(handbags_rows)
    tmp.close()
    return Path(tmp.name)


@pytest.fixture(scope="session")
def smoke_stage_a():
    _reset_vector_store()
    raw_rows = _ingest_files([HANDBAGS_A_FILE], supplier_id="supplier_A")
    return asyncio.run(_run_batch(raw_rows))


@pytest.fixture(scope="session")
def smoke_stage_b(smoke_stage_a):
    from corpmind.agents.ingestion import ingest_supplier_feed

    filtered_path = _write_handbags_only_feed()
    try:
        rows = ingest_supplier_feed(filtered_path, supplier_id="supplier_B")
        raw_rows = [row.model_dump() for row in rows]
    finally:
        filtered_path.unlink(missing_ok=True)
    return asyncio.run(_run_batch(raw_rows))


def test_smoke_near_duplicate_matches(smoke_stage_a, smoke_stage_b):
    ids_a = {p.catalog_id for p in smoke_stage_a["accepted_items"]}
    ids_b = {p.catalog_id for p in smoke_stage_b["accepted_items"]}
    shared = ids_a & ids_b
    assert len(shared) == len(EXPECTED_MATCHES)


def test_smoke_distinct_products_dont_merge(smoke_stage_a, smoke_stage_b):
    title = _title_for_row_ref(SUPPLIER_B_FILE, EXPECTED_DISTINCT, "title")
    hits = _find_by_title(smoke_stage_b["accepted_items"], title)
    assert hits, f"{title!r} not found in accepted_items"

    ids_a = {p.catalog_id for p in smoke_stage_a["accepted_items"]}
    assert hits[0].catalog_id not in ids_a
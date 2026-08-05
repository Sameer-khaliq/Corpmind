from pathlib import Path
import pytest
from corpmind.utils.batch_runner import run_batch
from corpmind.config import settings

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
GOLDSET_PATH = DATA_DIR / "gold_set" / "expected_outcomes.csv"

# Module-level skipif guard for CI/CD
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not GOLDSET_PATH.exists(),
        reason="Gold-set data files not available in CI environment"
    )
]

async def test_full_pipeline_shape(request):
    """
    Small batch (10-15 rows from the messy feed) through the REAL graph.
    Checks shape/invariants, not per-item correctness (trap_cases.py owns that).
    """
    # Lazy fixture resolution: Fixtures are fetched ONLY if the skip guard didn't trigger
    real_graph = request.getfixturevalue("real_graph")
    gold_set_rows = request.getfixturevalue("gold_set_rows")

    batch_rows = gold_set_rows[:15]
    result = await run_batch(real_graph, raw_rows=batch_rows, settings=settings)

    accepted = result["accepted_items"]
    flagged = result["flagged_items"]
    audit_log = result["audit_log"]

    # 1. No double-count regression (Day 17's bug)
    assert len(accepted) + len(flagged) == len(batch_rows)

    # 2. Every accepted item is a valid ConsistentProduct
    for item in accepted:
        assert item.catalog_id
        assert item.title

    # 3. Every accepted + flagged item has a traceable audit entry
    accepted_ids = {i.catalog_id for i in accepted}
    flagged_ids = {i.audit_catalog_id for i in flagged}
    audit_ids = {a.catalog_id for a in audit_log}
    assert accepted_ids.issubset(audit_ids)
    assert flagged_ids.issubset(audit_ids)

    # 4. Report generation doesn't error on real output
    from corpmind.agents.report import (
        generate_change_report, generate_catalog_export,
    )
    generate_change_report(accepted, flagged, audit_log)
    generate_catalog_export(accepted)
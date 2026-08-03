# tests/integration/test_pipeline_e2e.py
import pytest
from corpmind.utils.batch_runner import run_batch
from corpmind.config import settings

@pytest.mark.integration
async def test_full_pipeline_shape(real_graph, gold_set_rows):
    """
    Small batch (10-15 rows from the messy feed) through the REAL graph.
    Checks shape/invariants, not per-item correctness (trap_cases.py owns that).
    """
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
# tests/conftest.py
import pytest
from corpmind.graph.build_graph import build_graph

@pytest.fixture(scope="session")
def real_graph():
    """Real graph, real fns — no mocked judge/disambiguation/extraction."""
    return build_graph()  # default *_fn params, i.e. real ones

@pytest.fixture(scope="session")
def gold_set_rows():
    """
    Loads the labeled gold-set rows (Phase 0 dataset) that include
    the planted trap cases. TODO: confirm exact path/schema —
    I'm assuming data/gold_set/labeled_rows.json with fields
    {item_id, supplier_id, expected_decision, expected_group_id, notes}.
    Point me at the real file and I'll adjust the loader.
    """
    import json
    with open("data/gold_set/labeled_rows.json") as f:
        return json.load(f)

def rows_by_tag(gold_set_rows, tag: str):
    """Filter gold set rows by a 'trap_tag' field marking which trap case they belong to."""
    return [r for r in gold_set_rows if r.get("trap_tag") == tag]
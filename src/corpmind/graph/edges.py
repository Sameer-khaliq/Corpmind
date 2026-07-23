"""
graph/edges.py

CorpMind — Conditional edges (Day 14)
========================================

route_after_matching: the ONLY conditional edge named in the Day 14 spec.
Deterministic Python conditional on match_result.decision — no LLM
supervisor.

ROUTING DECISION, LOCKED IN (this replaced an earlier AMBIGUOUS-only-skip
version — if you have an older copy of this file lying around, delete it,
this is the one that matches the actual decision):
  - MATCHED_EXISTING -> enrich_and_evaluate  (only decision that gets Enrichment)
  - NEW_PRODUCT       -> evaluate_only        (evaluates extraction only)
  - AMBIGUOUS         -> evaluate_only        (runs Day 13's LLM disambiguation)
Deliberately keeps NEW_PRODUCT and AMBIGUOUS out of Enrichment — no existing
catalog entry to ground new-product enrichment against, and routing
AMBIGUOUS through Enrichment would build the complex cycle this decision
exists to avoid.
"""

from __future__ import annotations

from typing import Any

try:
    from langgraph.types import Send  # type: ignore
except ModuleNotFoundError:

    class Send:  # local stand-in — sandbox only
        def __init__(self, node: str, arg: Any):
            self.node = node
            self.arg = arg

        def __repr__(self) -> str:
            return f"Send(node={self.node!r})"

try:
    from corpmind.schemas.state import BatchState  # type: ignore
except ModuleNotFoundError:
    from graph.nodes import BatchState  # type: ignore


def route_after_matching(state: BatchState) -> list[Send]:
    """
    Fan out each item in matched_items (written once, sequentially, by
    phase_b_matching_node) based on its Phase B decision:
      - MATCHED_EXISTING -> enrich_and_evaluate
      - NEW_PRODUCT / AMBIGUOUS -> evaluate_only

    Both target nodes converge into the same `eval_out` accumulator key, so
    the join after this fan-out is implicit regardless of which branch each
    item took.
    """
    sends: list[Send] = []
    for item in state.get("matched_items", []):
        decision = (item.get("match_result") or {}).get("decision")
        if decision == "MATCHED_EXISTING":
            sends.append(Send("enrich_and_evaluate", item))
        else:  # NEW_PRODUCT or AMBIGUOUS
            sends.append(Send("evaluate_only", item))
    return sends


if __name__ == "__main__":
    fake_state: BatchState = {  # type: ignore[typeddict-item]
        "matched_items": [
            {"catalog_id": "cat-1", "match_result": {"decision": "NEW_PRODUCT"}},
            {"catalog_id": "cat-2", "match_result": {"decision": "MATCHED_EXISTING"}},
            {"catalog_id": None, "match_result": {"decision": "AMBIGUOUS", "catalog_id": None}},
        ]
    }
    routed = route_after_matching(fake_state)
    targets = [s.node for s in routed]
    assert targets == ["evaluate_only", "enrich_and_evaluate", "evaluate_only"], targets
    print("[edges] PASS — route_after_matching dispatches correctly:", targets)
    print("  (MATCHED_EXISTING -> enrich_and_evaluate; NEW_PRODUCT/AMBIGUOUS -> evaluate_only)")
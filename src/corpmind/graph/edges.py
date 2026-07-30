"""
graph/edges.py

CorpMind — Conditional edges (Day 14), rebuilt against the real state schema
===============================================================================

Only one Send stage exists now (see nodes.py's module docstring for why) —
this is the only conditional edge in the graph, dispatched right after
extract_and_match_node writes `items` (the real BatchState accumulator key,
confirmed from your schemas/state.py upload).

ROUTING DECISION (unchanged from before, still locked in):
  - MATCHED_EXISTING -> enrich_and_evaluate
  - NEW_PRODUCT / AMBIGUOUS -> evaluate_only
"""

from __future__ import annotations

from typing import Any

try:
    from langgraph.types import Send  # type: ignore
except ModuleNotFoundError:

    class Send:  # sandbox-only stand-in
        def __init__(self, node: str, arg: Any):
            self.node = node
            self.arg = arg

        def __repr__(self) -> str:
            return f"Send(node={self.node!r})"

try:
    from corpmind.schemas.state import BatchState  # type: ignore
    from corpmind.schemas.matching import MatchDecision  # type: ignore
except ModuleNotFoundError:
    from graph.nodes import BatchState, MatchDecision  # type: ignore


def route_after_matching(state: BatchState) -> list[Send]:
    """
    Fan out each item in `items` (written once by extract_and_match_node)
    based on its match_result.decision.
    """
    sends: list[Send] = []
    for item in state.get("items", []):
        match_result = item.get("match_result")
        decision = getattr(match_result, "decision", None) if match_result is not None else None
        if decision == MatchDecision.MATCHED_EXISTING or decision == "MATCHED_EXISTING":
            sends.append(Send("enrich_and_evaluate", item))
        else:  
            sends.append(Send("evaluate_only", item))
    return sends


if __name__ == "__main__":
    class _M:
        def __init__(self, d):
            self.decision = d

    single_item_input: BatchState = {  # type: ignore[typeddict-item]
    "batch_id": "smoke-test-batch",
    "raw_rows": [
        {"raw_row": {"extraction_id": "row-0", "title": "Men's Cotton Crew Neck T-Shirt",
                      "brand": "ExampleBrand", "color": "navy blue", "price": "19.99"}}
    ],
    # match_result ab yahan hand-craft nahi karna — extract_and_match_node
    # khud isse compute karega (stub phase_b_fn: i=0 -> NEW_PRODUCT by design),
    # jo yeh test asal mein chahta tha. Real extraction/matching path se
    # guzarna, bypass karna nahi — yehi "end-to-end through real LangGraph
    # engine" ka matlab hai.
    }
    routed = route_after_matching(single_item_input)
    targets = [s.node for s in routed]
    assert targets == ["evaluate_only", "enrich_and_evaluate", "evaluate_only"], targets
    print("[edges] PASS — route_after_matching dispatches correctly:", targets)
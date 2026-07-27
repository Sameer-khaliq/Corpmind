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
        else:  # NEW_PRODUCT or AMBIGUOUS
            sends.append(Send("evaluate_only", item))
    return sends


if __name__ == "__main__":
    class _M:
        def __init__(self, d):
            self.decision = d

    fake_state: BatchState = {  # type: ignore[typeddict-item]
        "items": [
            {"match_result": _M(MatchDecision.NEW_PRODUCT)},
            {"match_result": _M(MatchDecision.MATCHED_EXISTING)},
            {"match_result": _M(MatchDecision.AMBIGUOUS)},
        ]
    }
    routed = route_after_matching(fake_state)
    targets = [s.node for s in routed]
    assert targets == ["evaluate_only", "enrich_and_evaluate", "evaluate_only"], targets
    print("[edges] PASS — route_after_matching dispatches correctly:", targets)
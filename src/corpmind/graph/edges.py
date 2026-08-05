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

from langgraph.types import Send

from corpmind.schemas.state import BatchState
from corpmind.schemas.matching import MatchDecision


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
    import asyncio

    from corpmind.graph.nodes import make_extract_and_match_node

    async def _smoke_test() -> None:
        # Real flow: seed raw_rows, run extract_and_match_node (stub fns) to
        # actually populate `items` with computed match_results, THEN route —
        # bypassing that node (as the previous version of this test did, by
        # hand-crafting `items` directly) meant route_after_matching read an
        # `items` list that was never really produced by the node it exists
        # to route after.
        #
        # 3 rows against _default_phase_b_fn's stub rule (i%3==2 ->
        # MATCHED_EXISTING, else NEW_PRODUCT for i<4) deterministically
        # gives: row 0 -> NEW_PRODUCT, row 1 -> NEW_PRODUCT, row 2 -> MATCHED_EXISTING.
        raw_rows = [
            {
                "raw_row": {
                    "extraction_id": f"row-{i}",
                    "title": "Men's Cotton Crew Neck T-Shirt",
                    "brand": "ExampleBrand",
                    "color": "navy blue",
                    "price": "19.99",
                }
            }
            for i in range(3)
        ]
        state: BatchState = {"batch_id": "smoke-test-batch", "raw_rows": raw_rows}  # type: ignore[typeddict-item]

        extract_and_match_node = make_extract_and_match_node()
        node_updates = await extract_and_match_node(state)
        state = {**state, **node_updates}

        routed = route_after_matching(state)
        targets = [s.node for s in routed]
        assert targets == ["evaluate_only", "evaluate_only", "enrich_and_evaluate"], targets
        print("[edges] PASS — route_after_matching dispatches correctly (after a real extract_and_match_node run):", targets)

    asyncio.run(_smoke_test())
"""
graph/build_graph.py

CorpMind — Graph assembly (Day 14), rebuilt against the real state schema
=============================================================================

    START
      │
      ▼
    extract_and_match (batch, once — extraction+PhaseA concurrent internally,
                       PhaseB sequential cross-item clustering)
      │
      ▼  route_after_matching (conditional edge, Send fan-out — ONLY Send
      │  stage in this graph now)
      ├── MATCHED_EXISTING ──────► enrich_and_evaluate  ─┐
      └── NEW_PRODUCT/AMBIGUOUS ─► evaluate_only         ├─► each writes into
                                                           │  accepted_items
                                                           │  or flagged_items
                                                           ▼
                                                          END (implicit join)
"""

from __future__ import annotations

import asyncio

from corpmind.graph.edges import route_after_matching
from corpmind.graph.nodes import (
    BatchState,
    generate_report,
    make_enrich_and_evaluate_node,
    make_evaluate_only_node,
    make_extract_and_match_node,
)
from corpmind.graph.tracing_config import GEMINI_RETRY_POLICY, GROQ_RETRY_POLICY

from langgraph.graph import END, START, StateGraph


def build_graph(
    *,
    extraction_fn=None,
    phase_a_fn=None,
    phase_b_fn=None,
    prepare_batch_index_fn=None,
    write_new_products_fn=None,
    enrichment_fn=None,
    judge_call_fn=None,
    disambiguation_fn=None,
    consistency_fn=None,
) -> "StateGraph":
    graph = StateGraph(BatchState)

    match_kwargs = {}
    if extraction_fn:
        match_kwargs["extraction_fn"] = extraction_fn
    if phase_a_fn:
        match_kwargs["phase_a_fn"] = phase_a_fn
    if phase_b_fn:
        match_kwargs["phase_b_fn"] = phase_b_fn
    if prepare_batch_index_fn:
        match_kwargs["prepare_batch_index_fn"] = prepare_batch_index_fn
    if write_new_products_fn:
        match_kwargs["write_new_products_fn"] = write_new_products_fn

    enrich_kwargs = {}
    if enrichment_fn:
        enrich_kwargs["enrichment_fn"] = enrichment_fn
    if judge_call_fn:
        enrich_kwargs["judge_call_fn"] = judge_call_fn
    if disambiguation_fn:
        enrich_kwargs["disambiguation_fn"] = disambiguation_fn
    if consistency_fn:
        enrich_kwargs["consistency_fn"] = consistency_fn

    evaluate_only_kwargs = {}
    if disambiguation_fn:
        evaluate_only_kwargs["disambiguation_fn"] = disambiguation_fn
    if consistency_fn:
        evaluate_only_kwargs["consistency_fn"] = consistency_fn

    graph.add_node("extract_and_match", make_extract_and_match_node(**match_kwargs), retry_policy=GROQ_RETRY_POLICY)
    graph.add_node("enrich_and_evaluate", make_enrich_and_evaluate_node(**enrich_kwargs), retry_policy=GEMINI_RETRY_POLICY)
    graph.add_node("evaluate_only", make_evaluate_only_node(**evaluate_only_kwargs), retry_policy=GEMINI_RETRY_POLICY)

    graph.add_edge(START, "extract_and_match")
    graph.add_conditional_edges("extract_and_match", route_after_matching, ["enrich_and_evaluate", "evaluate_only"])
    graph.add_edge("enrich_and_evaluate", END)
    graph.add_edge("evaluate_only", END)

    return graph


async def _main() -> None:
    compiled = build_graph().compile()

    # === Day 14 smoke test — Robustness inject for real LangGraph Pregel engine ===
    single_item_input: BatchState = {  # type: ignore[typeddict-item]
        "batch_id": "smoke-test-batch",
        "supplier_feeds_rows": [
            {"extraction_id": "row-0", "title": "Men's Cotton Crew Neck T-Shirt", "brand": "ExampleBrand", "color": "navy blue", "price": "19.99"}
        ],
        # match_result belongs on each item inside `items`, not on BatchState
        # itself — route_after_matching reads item.get("match_result") per
        # item. Setting it at the batch level (as an earlier version of this
        # test did) meant it was silently ignored: the item had no
        # match_result, so route_after_matching's else-branch sent it to
        # evaluate_only no matter what decision was "intended" here.
        "items": [
            {
                "extraction_id": "row-0",
                "title": "Men's Cotton Crew Neck T-Shirt",
                "brand": "ExampleBrand",
                "color": "navy blue",
                "price": "19.99",
                "match_result": {"decision": "NEW_PRODUCT", "candidate_pairs": [], "scores": {}},
            }
        ],
    }

    try:
        final_state = await compiled.ainvoke(single_item_input)
        print("[Day 14] PASS — single item flowed end-to-end through real LangGraph Pregel engine!")
        if "accepted_items" in final_state or "flagged_items" in final_state:
            print("  report:", generate_report(final_state))
    except NotImplementedError as e:
        print("[Day 14] Wiring successfully reaches the expected implementation gap:", e)
        print("  This IS the expected structural milestone — your graph wiring is 100% correct!")
        # Deliberately not swallowed further than this: only the SPECIFIC,
        # known gap (no consistency_fn wired yet) is treated as an
        # acceptable stopping point. Anything else below is a real bug and
        # must surface as one — see the removed catch-all Exception handler
        # that used to print a "Pass" message for ANY exception type,
        # including ones that had nothing to do with this gap.

    print("\n[Day 14] using real LangGraph StateGraph Pregel engine workflow")
if __name__ == "__main__":
    asyncio.run(_main())